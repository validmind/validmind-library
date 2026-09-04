# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Small, file-backed OIDC credential store used by the tracking SDKs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
    key = credential_key(issuer, client_id, audience)
    normalized_issuer = normalize_issuer(issuer)
    normalized_audience = normalize_audience(audience)
    data = load_credentials_file(path)
    credentials = dict(data.get("credentials", {}))
    row = {"issuer": normalized_issuer, "client_id": client_id, **entry}
    if normalized_audience:
        row["audience"] = normalized_audience
    credentials[key] = row
    data["credentials"] = credentials
    save_credentials_file(data, path)


def delete_cached_entry(
    issuer: str,
    client_id: str,
    path: Optional[Path] = None,
    audience: Optional[str] = None,
) -> None:
    data = load_credentials_file(path)
    credentials = dict(data.get("credentials", {}))
    credentials.pop(credential_key(issuer, client_id, audience), None)
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
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
