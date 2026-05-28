# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Persist OIDC tokens for library authentication under ``~/.validmind/``."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ValidMindAuthError

_CREDENTIALS_VERSION = 1


def normalize_issuer(issuer: str) -> str:
    """Strip whitespace, trailing slash, and matching outer quotes.

    Surrounding quotes often appear when copying from ``.env`` or notebooks
    (e.g. ``'https://idp.example.com'``), which would otherwise break HTTP clients.
    """
    base = issuer.strip().rstrip("/")
    while len(base) >= 2 and base[0] == base[-1] and base[0] in ('"', "'"):
        base = base[1:-1].strip().rstrip("/")
    return base


def normalize_client_id(client_id: str) -> str:
    """Strip whitespace and matching outer quotes from OAuth ``client_id``."""
    base = client_id.strip()
    while len(base) >= 2 and base[0] == base[-1] and base[0] in ('"', "'"):
        base = base[1:-1].strip()
    return base


def normalize_audience(audience: Optional[str]) -> str:
    """Normalize OAuth resource/API ``audience`` (Identifier). Empty if unset."""
    if not audience:
        return ""
    base = audience.strip()
    while len(base) >= 2 and base[0] == base[-1] and base[0] in ('"', "'"):
        base = base[1:-1].strip()
    return base


def credential_key(issuer: str, client_id: str, audience: Optional[str] = None) -> str:
    base = f"{normalize_issuer(issuer)}|{normalize_client_id(client_id)}"
    aud = normalize_audience(audience)
    if aud:
        return f"{base}|{aud}"
    return base


def credentials_path() -> Path:
    return Path.home() / ".validmind" / "credentials.json"


def _empty_store() -> Dict[str, Any]:
    return {"version": _CREDENTIALS_VERSION, "credentials": {}}


def load_credentials_file(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or credentials_path()
    if not path.is_file():
        return _empty_store()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValidMindAuthError(f"Could not read credentials file {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValidMindAuthError(f"Invalid credentials file format at {path}")
    data.setdefault("version", _CREDENTIALS_VERSION)
    data.setdefault("credentials", {})
    return data


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=".credentials-", suffix=".tmp", text=True
    )
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def save_credentials_file(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or credentials_path()
    data = dict(data)
    data["version"] = _CREDENTIALS_VERSION
    if "credentials" not in data or not isinstance(data["credentials"], dict):
        data["credentials"] = {}
    _atomic_write(path, data)


def get_cached_entry(
    issuer: str,
    client_id: str,
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    key = credential_key(issuer, client_id, audience)
    data = load_credentials_file(path)
    entry = data.get("credentials", {}).get(key)
    if not entry:
        return None
    return dict(entry)


def upsert_cached_entry(
    issuer: str,
    client_id: str,
    entry: Dict[str, Any],
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> None:
    key = credential_key(issuer, client_id, audience)
    norm_issuer = normalize_issuer(issuer)
    aud = normalize_audience(audience)
    data = load_credentials_file(path)
    credentials = dict(data.get("credentials", {}))
    row = {
        "issuer": norm_issuer,
        "client_id": client_id,
        **entry,
    }
    if aud:
        row["audience"] = aud
    credentials[key] = row
    data["credentials"] = credentials
    save_credentials_file(data, path)


def delete_cached_entry(
    issuer: str,
    client_id: str,
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> None:
    key = credential_key(issuer, client_id, audience)
    data = load_credentials_file(path)
    credentials = dict(data.get("credentials", {}))
    credentials.pop(key, None)
    data["credentials"] = credentials
    save_credentials_file(data, path)


def is_expired(entry: Dict[str, Any], skew_seconds: int = 120) -> bool:
    raw = entry.get("expires_at")
    if not raw:
        return True
    try:
        expires = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires - timedelta(seconds=skew_seconds)


def expires_at_from_secs(expires_in: Optional[int]) -> str:
    seconds = int(expires_in) if expires_in is not None else 3600
    when = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return when.isoformat()
