# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Pluggable OIDC token persistence for library authentication."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .errors import ValidMindAuthError

_ENV_BACKEND_SPEC = "VM_OIDC_CREDENTIALS_BACKEND"
_ENV_BACKEND_KWARGS = "VM_OIDC_CREDENTIALS_BACKEND_KWARGS"
_ENV_NO_PERSIST = "VM_OIDC_NO_PERSIST"

_default_backend: Optional["OidcCredentialsBackend"] = None


@runtime_checkable
class OidcCredentialsBackend(Protocol):
    """Persist OIDC token entries keyed by :func:`credentials_store.credential_key`."""

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a cached token entry or ``None``."""
        ...

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        """Persist a token entry."""
        ...

    def delete(self, key: str) -> None:
        """Remove a cached token entry."""
        ...


def set_default_backend(backend: Optional[OidcCredentialsBackend]) -> None:
    """Register a process-wide default backend (e.g. from IPython startup)."""
    global _default_backend
    _default_backend = backend


def get_default_backend() -> Optional[OidcCredentialsBackend]:
    """Return the registered process-wide default backend, if any."""
    return _default_backend


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _validate_backend(backend: Any, spec: str) -> OidcCredentialsBackend:
    if isinstance(backend, OidcCredentialsBackend):
        return backend
    for method in ("get", "put", "delete"):
        if not callable(getattr(backend, method, None)):
            raise ValidMindAuthError(
                f"OIDC credentials backend {spec!r} must implement get, put, and delete"
            )
    return backend  # type: ignore[return-value]


def load_backend_from_env() -> Optional[OidcCredentialsBackend]:
    """Instantiate a backend from ``VM_OIDC_*`` environment variables."""
    if _truthy_env(_ENV_NO_PERSIST):
        return MemoryCredentialsBackend()
    spec = os.getenv(_ENV_BACKEND_SPEC, "").strip()
    if not spec:
        return None
    return _instantiate_backend(spec)


def _instantiate_backend(spec: str) -> OidcCredentialsBackend:
    module_path, sep, class_name = spec.partition(":")
    if not sep:
        raise ValidMindAuthError(
            "VM_OIDC_CREDENTIALS_BACKEND must be module.path:ClassName, "
            f"got {spec!r}"
        )
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValidMindAuthError(
            f"Could not load OIDC credentials backend {spec!r}: {exc}"
        ) from exc

    kwargs: Dict[str, Any] = {}
    raw_kwargs = os.getenv(_ENV_BACKEND_KWARGS, "").strip()
    if raw_kwargs:
        try:
            parsed = json.loads(raw_kwargs)
        except json.JSONDecodeError as exc:
            raise ValidMindAuthError(
                f"VM_OIDC_CREDENTIALS_BACKEND_KWARGS is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValidMindAuthError(
                "VM_OIDC_CREDENTIALS_BACKEND_KWARGS must be a JSON object"
            )
        kwargs = parsed

    try:
        backend = cls(**kwargs)
    except TypeError as exc:
        raise ValidMindAuthError(
            f"Could not construct OIDC credentials backend {spec!r}: {exc}"
        ) from exc

    return _validate_backend(backend, spec)


def resolve_credentials_backend(
    explicit: Optional[OidcCredentialsBackend] = None,
) -> OidcCredentialsBackend:
    """
    Resolve the active credentials backend.

    Precedence: explicit argument → :func:`set_default_backend` → environment →
    :class:`FileCredentialsBackend`.
    """
    if explicit is not None:
        return explicit
    if _default_backend is not None:
        return _default_backend
    from_env = load_backend_from_env()
    if from_env is not None:
        return from_env
    return FileCredentialsBackend()


class MemoryCredentialsBackend:
    """In-process token store; nothing is written to disk."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._entries.get(key)
        return dict(entry) if entry else None

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        self._entries[key] = dict(entry)

    def delete(self, key: str) -> None:
        self._entries.pop(key, None)


# Explicit alias for documentation and env-based registration.
NullCredentialsBackend = MemoryCredentialsBackend


class FileCredentialsBackend:
    """Default backend: ``~/.validmind/credentials.json`` with mode ``0600``."""

    def __init__(self, path: Optional[Path] = None) -> None:
        from .credentials_store import credentials_path

        self._path = path or credentials_path()

    @property
    def path(self) -> Path:
        return self._path

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        from .credentials_store import load_credentials_file

        data = load_credentials_file(self._path)
        entry = data.get("credentials", {}).get(key)
        return dict(entry) if entry else None

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        from .credentials_store import load_credentials_file, save_credentials_file

        data = load_credentials_file(self._path)
        credentials = dict(data.get("credentials", {}))
        credentials[key] = dict(entry)
        data["credentials"] = credentials
        save_credentials_file(data, self._path)

    def delete(self, key: str) -> None:
        from .credentials_store import load_credentials_file, save_credentials_file

        data = load_credentials_file(self._path)
        credentials = dict(data.get("credentials", {}))
        credentials.pop(key, None)
        data["credentials"] = credentials
        save_credentials_file(data, self._path)
