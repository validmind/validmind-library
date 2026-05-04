# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""OIDC device authorization flow (RFC 8628) for notebook-style login."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import requests
from requests import Response

from .credentials_store import (
    expires_at_from_secs,
    normalize_audience,
    normalize_issuer,
)
from .errors import ValidMindAuthError

_OPENID_CONFIG_SUFFIX = "/.well-known/openid-configuration"
_DEFAULT_TIMEOUT = 30.0

_config_cache: Dict[str, Dict[str, Any]] = {}


def _print_device_authorization_prompt(verification_uri: str, user_code: str) -> None:
    """Print RFC 8628 verification instructions for interactive login.

    ``verification_uri`` and ``user_code`` are intended for human-readable display;
    they are not authentication secrets (see RFC 8628 § 3.3). The ``device_code``
    binding must remain confidential and is never written here.
    """
    msg = (
        f"Visit: {verification_uri}\n"
        f"Code:  {user_code}\n"
        "Waiting for authorization..."
    )
    print(msg)  # lgtm[py/clear-text-logging-sensitive-data]


def fetch_openid_configuration(
    issuer: str, timeout: float = _DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """GET OpenID Provider configuration document."""
    base = normalize_issuer(issuer)
    if base in _config_cache:
        return _config_cache[base]
    url = f"{base}{_OPENID_CONFIG_SUFFIX}"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise ValidMindAuthError(
            f"Could not reach OIDC discovery URL {url!r}: {e}"
        ) from e
    if r.status_code != 200:
        raise ValidMindAuthError(
            f"OIDC discovery failed for {url!r}: HTTP {r.status_code} {r.text[:500]}"
        )
    try:
        cfg = r.json()
    except ValueError as e:
        raise ValidMindAuthError(
            f"OIDC discovery returned non-JSON from {url!r}"
        ) from e
    for key in ("device_authorization_endpoint", "token_endpoint"):
        if key not in cfg:
            raise ValidMindAuthError(
                f"OIDC discovery document from {url!r} is missing {key!r}"
            )
    _config_cache[base] = cfg
    return cfg


def request_device_authorization(
    device_authorization_endpoint: str,
    client_id: str,
    scope: str,
    timeout: float = _DEFAULT_TIMEOUT,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {"client_id": client_id, "scope": scope}
    aud = normalize_audience(audience)
    if aud:
        payload["audience"] = aud
    try:
        r = requests.post(
            device_authorization_endpoint,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise ValidMindAuthError(f"Device authorization request failed: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        raise ValidMindAuthError(
            "Device authorization endpoint rejected the request: "
            f"HTTP {r.status_code} {body or r.text[:500]}"
        )
    for key in ("device_code", "user_code", "verification_uri"):
        if key not in body:
            raise ValidMindAuthError(
                f"Device authorization response missing {key!r}: {body}"
            )
    return body


def _post_device_token_poll(
    token_endpoint: str,
    client_id: str,
    device_code: str,
    *,
    timeout: float,
    audience: Optional[str],
) -> Tuple[Response, Dict[str, Any]]:
    """POST once to the token endpoint for device-code grant; returns response and JSON body."""
    token_body: Dict[str, str] = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code,
        "client_id": client_id,
    }
    aud = normalize_audience(audience)
    if aud:
        token_body["audience"] = aud
    try:
        r = requests.post(
            token_endpoint,
            data=token_body,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise ValidMindAuthError(f"Token poll request failed: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    return r, body


def _handle_device_token_poll_response(
    r: Response,
    body: Dict[str, Any],
    current_interval: float,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Map poll response to success payload, retry with updated interval, or raise."""
    if r.status_code == 200 and "access_token" in body:
        return body, current_interval

    error = body.get("error")
    if error == "authorization_pending":
        time.sleep(current_interval)
        return None, current_interval
    if error == "slow_down":
        current_interval += 5
        time.sleep(current_interval)
        return None, current_interval
    if error == "expired_token":
        raise ValidMindAuthError(
            "Device login expired before completion. Run vm.init() again to start a new login."
        )
    if error == "access_denied":
        raise ValidMindAuthError("Device authorization was denied.")
    raise ValidMindAuthError(
        f"Token poll failed: HTTP {r.status_code} error={error!r} {body or r.text[:500]}"
    )


def poll_device_token(
    token_endpoint: str,
    client_id: str,
    device_code: str,
    *,
    interval: float = 5.0,
    expires_in: float = 900.0,
    timeout: float = _DEFAULT_TIMEOUT,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    """Poll token endpoint until success or terminal OAuth error."""
    deadline = time.monotonic() + float(expires_in)
    current_interval = float(interval)

    while time.monotonic() < deadline:
        r, body = _post_device_token_poll(
            token_endpoint,
            client_id,
            device_code,
            timeout=timeout,
            audience=audience,
        )
        token_payload, current_interval = _handle_device_token_poll_response(
            r, body, current_interval
        )
        if token_payload is not None:
            return token_payload

    raise ValidMindAuthError(
        "Device login timed out waiting for authorization. Run vm.init() again."
    )


def refresh_access_token(
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
    scope: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    data: Dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if scope:
        data["scope"] = scope
    aud = normalize_audience(audience)
    if aud:
        data["audience"] = aud
    try:
        r = requests.post(
            token_endpoint,
            data=data,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise ValidMindAuthError(f"Token refresh request failed: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200 or "access_token" not in body:
        raise ValidMindAuthError(
            f"Token refresh failed: HTTP {r.status_code} {body or r.text[:500]}"
        )
    return body


def run_device_flow(
    issuer: str,
    client_id: str,
    scope: str,
    *,
    audience: Optional[str] = None,
    status_callback=None,
) -> Dict[str, Any]:
    """
    Run full RFC 8628 device authorization flow.

    ``audience`` is the OAuth resource identifier (e.g. Auth0 API Identifier). When
    set, providers such as Auth0 typically return an RS256 access token for that API.

    ``status_callback`` receives dict milestones (optional); default UX prints instructions.
    """
    cfg = fetch_openid_configuration(issuer)
    dev = request_device_authorization(
        cfg["device_authorization_endpoint"],
        client_id,
        scope,
        audience=audience,
    )

    verification_uri = dev["verification_uri"]
    user_code = dev["user_code"]
    if status_callback:
        status_callback(
            {
                "verification_uri": verification_uri,
                "user_code": user_code,
                "verification_uri_complete": dev.get("verification_uri_complete"),
            }
        )
    else:
        _print_device_authorization_prompt(verification_uri, user_code)

    interval = float(dev.get("interval", 5))
    expires_in = float(dev.get("expires_in", 900))
    token_payload = poll_device_token(
        cfg["token_endpoint"],
        client_id,
        dev["device_code"],
        interval=interval,
        expires_in=expires_in,
        audience=audience,
    )

    access_token = token_payload["access_token"]
    refresh_tok = token_payload.get("refresh_token")
    id_token = token_payload.get("id_token")
    expires_at = expires_at_from_secs(token_payload.get("expires_in"))

    aud_norm = normalize_audience(audience)
    out: Dict[str, Any] = {
        "issuer": normalize_issuer(issuer),
        "client_id": client_id,
        "access_token": access_token,
        "refresh_token": refresh_tok,
        "id_token": id_token,
        "expires_at": expires_at,
    }
    if aud_norm:
        out["audience"] = aud_norm
    return out


def try_refresh_cached_tokens(
    issuer: str,
    client_id: str,
    refresh_token: str,
    scope: Optional[str],
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = fetch_openid_configuration(issuer)
    refreshed = refresh_access_token(
        cfg["token_endpoint"],
        client_id,
        refresh_token,
        scope=scope,
        audience=audience,
    )
    new_refresh = refreshed.get("refresh_token") or refresh_token
    aud_norm = normalize_audience(audience)
    out: Dict[str, Any] = {
        "issuer": normalize_issuer(issuer),
        "client_id": client_id,
        "access_token": refreshed["access_token"],
        "refresh_token": new_refresh,
        "id_token": refreshed.get("id_token"),
        "expires_at": expires_at_from_secs(refreshed.get("expires_in")),
    }
    if aud_norm:
        out["audience"] = aud_norm
    return out


def clear_configuration_cache() -> None:
    """Test helper: drop cached discovery documents."""
    _config_cache.clear()
