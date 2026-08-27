# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Synchronous OIDC device-flow authentication for tracking clients."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

import requests

from .credentials_store import (
    delete_cached_entry,
    expires_at_from_secs,
    get_cached_entry,
    is_expired,
    normalize_audience,
    normalize_client_id,
    normalize_issuer,
    upsert_cached_entry,
)
from .errors import TrackingAuthError

_OPENID_CONFIG_SUFFIX = "/.well-known/openid-configuration"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_SCOPE = "openid profile email offline_access"


def _token_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = dict(payload)
    if not entry.get("expires_at"):
        entry["expires_at"] = expires_at_from_secs(entry.get("expires_in"))
    return entry


def _bearer_token(entry: Dict[str, Any]) -> str:
    if "login.microsoftonline.com" in entry.get("issuer", "").lower() and entry.get(
        "id_token"
    ):
        return entry["id_token"]
    token = entry.get("access_token")
    if not token:
        raise TrackingAuthError("OIDC response did not contain an access token")
    return token


def _response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = {}
    return body if isinstance(body, dict) else {}


def fetch_openid_configuration(
    issuer: str, timeout: float = _DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    base = normalize_issuer(issuer)
    url = f"{base}{_OPENID_CONFIG_SUFFIX}"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise TrackingAuthError(
            f"Could not reach OIDC discovery URL {url!r}: {exc}"
        ) from exc
    if response.status_code != 200:
        raise TrackingAuthError(
            f"OIDC discovery failed for {url!r}: HTTP {response.status_code} "
            f"{response.text[:500]}"
        )
    body = _response_json(response)
    for key in ("device_authorization_endpoint", "token_endpoint"):
        if key not in body:
            raise TrackingAuthError(
                f"OIDC discovery document from {url!r} is missing {key!r}"
            )
    return body


def request_device_authorization(
    endpoint: str,
    client_id: str,
    scope: str,
    timeout: float = _DEFAULT_TIMEOUT,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, str] = {"client_id": client_id, "scope": scope}
    normalized_audience = normalize_audience(audience)
    if normalized_audience:
        payload["audience"] = normalized_audience
    try:
        response = requests.post(
            endpoint,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TrackingAuthError(f"Device authorization request failed: {exc}") from exc
    body = _response_json(response)
    if response.status_code != 200:
        raise TrackingAuthError(
            "Device authorization endpoint rejected the request: "
            f"HTTP {response.status_code} {body or response.text[:500]}"
        )
    for key in ("device_code", "user_code", "verification_uri"):
        if key not in body:
            raise TrackingAuthError(f"Device authorization response missing {key!r}")
    return body


def poll_device_token(
    endpoint: str,
    client_id: str,
    device_code: str,
    *,
    interval: float = 5.0,
    expires_in: float = 900.0,
    timeout: float = _DEFAULT_TIMEOUT,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    deadline = time.monotonic() + float(expires_in)
    current_interval = float(interval)
    while time.monotonic() < deadline:
        payload: Dict[str, str] = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": client_id,
        }
        normalized_audience = normalize_audience(audience)
        if normalized_audience:
            payload["audience"] = normalized_audience
        try:
            response = requests.post(
                endpoint,
                data=payload,
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise TrackingAuthError(f"Token poll request failed: {exc}") from exc
        body = _response_json(response)
        if response.status_code == 200 and body.get("access_token"):
            return _token_entry(body)
        error = body.get("error")
        if error == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error == "slow_down":
            current_interval += 5
            time.sleep(current_interval)
            continue
        if error == "expired_token":
            raise TrackingAuthError("Device login expired before completion")
        if error == "access_denied":
            raise TrackingAuthError("Device authorization was denied")
        raise TrackingAuthError(
            f"Token poll failed: HTTP {response.status_code} "
            f"error={error!r} {body or response.text[:500]}"
        )
    raise TrackingAuthError("Device login timed out waiting for authorization")


def refresh_access_token(
    endpoint: str,
    client_id: str,
    refresh_token: str,
    scope: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if scope:
        payload["scope"] = scope
    normalized_audience = normalize_audience(audience)
    if normalized_audience:
        payload["audience"] = normalized_audience
    try:
        response = requests.post(
            endpoint,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TrackingAuthError(f"Token refresh request failed: {exc}") from exc
    body = _response_json(response)
    if response.status_code != 200 or not body.get("access_token"):
        raise TrackingAuthError(
            f"Token refresh failed: HTTP {response.status_code} "
            f"{body or response.text[:500]}"
        )
    return _token_entry(body)


def run_device_flow(
    issuer: str,
    client_id: str,
    scope: str,
    *,
    audience: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    configuration = fetch_openid_configuration(issuer, timeout=timeout)
    device = request_device_authorization(
        configuration["device_authorization_endpoint"],
        client_id,
        scope,
        timeout=timeout,
        audience=audience,
    )
    status = {
        "verification_uri": device["verification_uri"],
        "user_code": device["user_code"],
        "verification_uri_complete": device.get("verification_uri_complete"),
    }
    if status_callback:
        status_callback(status)
    else:
        complete = status.get("verification_uri_complete") or status["verification_uri"]
        print(
            f"Visit: {complete}\nCode:  {status['user_code']}\nWaiting for authorization..."
        )
    return poll_device_token(
        configuration["token_endpoint"],
        client_id,
        device["device_code"],
        interval=float(device.get("interval", 5)),
        expires_in=float(device.get("expires_in", 900)),
        timeout=timeout,
        audience=audience,
    )


class OIDCAuthenticator:
    """Load, refresh, or interactively obtain a bearer token."""

    def __init__(
        self,
        issuer: str,
        client_id: str,
        *,
        scope: Optional[str] = None,
        audience: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        credentials_path_value=None,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.issuer = normalize_issuer(issuer)
        self.client_id = normalize_client_id(client_id)
        self.scope = scope or _DEFAULT_SCOPE
        self.audience = normalize_audience(audience) or None
        self.timeout = timeout
        self.credentials_path = credentials_path_value
        self.status_callback = status_callback
        self._entry: Optional[Dict[str, Any]] = None
        self._refresh_lock = threading.Lock()

    def initialize(self) -> None:
        cached = get_cached_entry(
            self.issuer,
            self.client_id,
            path=self.credentials_path,
            audience=self.audience,
        )
        if cached and not is_expired(cached):
            self._entry = cached
            return
        if cached and cached.get("refresh_token"):
            try:
                refreshed = refresh_access_token(
                    self._token_endpoint(cached),
                    self.client_id,
                    cached["refresh_token"],
                    scope=self.scope,
                    timeout=self.timeout,
                    audience=self.audience,
                )
            except TrackingAuthError:
                delete_cached_entry(
                    self.issuer,
                    self.client_id,
                    path=self.credentials_path,
                    audience=self.audience,
                )
            else:
                refreshed.setdefault("refresh_token", cached["refresh_token"])
                self._save(refreshed)
                return
        entry = run_device_flow(
            self.issuer,
            self.client_id,
            self.scope,
            audience=self.audience,
            timeout=self.timeout,
            status_callback=self.status_callback,
        )
        self._save(entry)

    def token(self) -> str:
        if self._entry and not is_expired(self._entry):
            return _bearer_token(self._entry)
        with self._refresh_lock:
            if self._entry and not is_expired(self._entry):
                return _bearer_token(self._entry)
            cached = get_cached_entry(
                self.issuer,
                self.client_id,
                path=self.credentials_path,
                audience=self.audience,
            )
            if not cached or not cached.get("refresh_token"):
                raise TrackingAuthError(
                    "OIDC access token is missing or expired; initialize the client again"
                )
            refreshed = refresh_access_token(
                self._token_endpoint(cached),
                self.client_id,
                cached["refresh_token"],
                scope=self.scope,
                timeout=self.timeout,
                audience=self.audience,
            )
            refreshed.setdefault("refresh_token", cached["refresh_token"])
            self._save(refreshed)
            return _bearer_token(self._entry)

    def _save(self, entry: Dict[str, Any]) -> None:
        saved = _token_entry(entry)
        saved["issuer"] = self.issuer
        saved["client_id"] = self.client_id
        self._entry = saved
        upsert_cached_entry(
            self.issuer,
            self.client_id,
            saved,
            path=self.credentials_path,
            audience=self.audience,
        )

    def _token_endpoint(self, entry: Dict[str, Any]) -> str:
        configuration = fetch_openid_configuration(self.issuer, timeout=self.timeout)
        return configuration["token_endpoint"]
