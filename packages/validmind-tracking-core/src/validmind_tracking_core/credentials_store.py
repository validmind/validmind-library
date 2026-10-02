# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Small, file-backed OIDC credential store used by the tracking SDKs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

from .errors import TrackingAuthError

_CREDENTIALS_VERSION = 1


def normalize_issuer(issuer: str) -> str:
    base = issuer.strip().rstrip("/")
    while len(base) >= 2 and base[0] == base[-1] and base[0] in ('"', "'"):
        base = base[1:-1].strip().rstrip("/")
    return base


def normalize_client_id(client_id: str) -> str:
    base = client_id.strip()
    while len(base) >= 2 and base[0] == base[-1] and base[0] in ('"', "'"):
        base = base[1:-1].strip()
    return base


def normalize_audience(audience: Optional[str]) -> str:
    if not audience:
        return ""
    base = audience.strip()
    while len(base) >= 2 and base[0] == base[-1] and base[0] in ('"', "'"):
        base = base[1:-1].strip()
    return base


def credential_key(issuer: str, client_id: str, audience: Optional[str] = None) -> str:
    base = f"{normalize_issuer(issuer)}|{normalize_client_id(client_id)}"
    aud = normalize_audience(audience)
    return f"{base}|{aud}" if aud else base


def credentials_path() -> Path:
    return Path.home() / ".validmind" / "credentials.json"


def _empty_store() -> Dict[str, Any]:
    return {"version": _CREDENTIALS_VERSION, "credentials": {}}


def load_credentials_file(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or credentials_path()
    if not path.is_file():
        return _empty_store()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise TrackingAuthError(
            f"Could not read credentials file {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise TrackingAuthError(f"Invalid credentials file format at {path}")
    data.setdefault("version", _CREDENTIALS_VERSION)
    data.setdefault("credentials", {})
    return data


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".credentials-", suffix=".tmp", text=True
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def save_credentials_file(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or credentials_path()
    normalized = dict(data)
    normalized["version"] = _CREDENTIALS_VERSION
    if not isinstance(normalized.get("credentials"), dict):
        normalized["credentials"] = {}
    _atomic_write(path, normalized)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock across a read-modify-write of the credentials file.

    A sidecar lock file is used because the credentials file itself is replaced
    atomically on every write.
    """
    # ponytail: no cross-process lock on Windows (no fcntl); add msvcrt.locking if
    # multi-worker Windows services need it.
    if fcntl is None:
        yield
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def get_cached_entry(
    issuer: str,
    client_id: str,
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    key = credential_key(issuer, client_id, audience)
    entry = load_credentials_file(path).get("credentials", {}).get(key)
    return dict(entry) if entry else None


def upsert_cached_entry(
    issuer: str,
    client_id: str,
    entry: Dict[str, Any],
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> None:
    path = path or credentials_path()
    key = credential_key(issuer, client_id, audience)
    row = {"issuer": normalize_issuer(issuer), "client_id": client_id, **entry}
    normalized_audience = normalize_audience(audience)
    if normalized_audience:
        row["audience"] = normalized_audience
    with _locked(path):
        data = load_credentials_file(path)
        credentials = dict(data.get("credentials", {}))
        credentials[key] = row
        data["credentials"] = credentials
        save_credentials_file(data, path)


def delete_cached_entry(
    issuer: str,
    client_id: str,
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> None:
    path = path or credentials_path()
    with _locked(path):
        data = load_credentials_file(path)
        credentials = dict(data.get("credentials", {}))
        credentials.pop(credential_key(issuer, client_id, audience), None)
        data["credentials"] = credentials
        save_credentials_file(data, path)


def _parse_timestamp(raw: str) -> datetime:
    # datetime.fromisoformat before Python 3.11 accepts only 3 or 6 fractional
    # digits and "+HH:MM" offsets, so normalize "Z", nanoseconds and "+HHMM".
    raw = raw.strip().replace("Z", "+00:00")
    raw = re.sub(r"\.(\d+)", lambda m: "." + m.group(1)[:6].ljust(6, "0"), raw)
    raw = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw)
    return datetime.fromisoformat(raw)


def is_expired(entry: Dict[str, Any], skew_seconds: int = 120) -> bool:
    raw = entry.get("expires_at")
    if not raw:
        return True
    try:
        expires = _parse_timestamp(raw)
    except (TypeError, ValueError, AttributeError):
        return True
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires - timedelta(seconds=skew_seconds)


def expires_at_from_secs(expires_in: Optional[int]) -> str:
    seconds = int(expires_in) if expires_in is not None else 3600
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
