# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Dependency-light synchronous and async-compatible metric transport."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urljoin

import requests

from .errors import TrackingAPIError, TrackingConfigurationError
from .oidc import OIDCAuthenticator


def _validate_metric(
    key: str,
    value: Any,
    thresholds: Optional[Dict[str, Any]],
) -> None:
    if not key or not isinstance(key, str):
        raise ValueError("`key` must be a non-empty string")
    if value is None:
        raise ValueError("Must provide a value for the metric")
    if not isinstance(value, (int, float)):
        raise ValueError(
            "Only scalar values (int or float) are allowed for logging metrics."
        )
    if thresholds is not None and not isinstance(thresholds, dict):
        raise ValueError("`thresholds` must be a dictionary or None")


def serialize_metric(
    key: str,
    value: Any,
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    passed: Optional[bool] = None,
    *,
    encoder: Optional[Type[json.JSONEncoder]] = None,
) -> str:
    """Validate and serialize a metric using the tracking API schema."""
    _validate_metric(key, value, thresholds)
    payload = {
        "key": key,
        "value": value,
        "inputs": inputs or [],
        "params": params or {},
        "recorded_at": recorded_at,
        "thresholds": thresholds or {},
        "passed": passed if passed is not None else None,
    }
    kwargs = {"allow_nan": False}
    if encoder is not None:
        kwargs["cls"] = encoder
    return json.dumps(payload, **kwargs)


def post_metric(
    url: str,
    body: str,
    headers: Dict[str, str],
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """POST a serialized metric and return the JSON response."""
    response = requests.post(url, data=body, headers=headers, timeout=timeout)
    if response.status_code != 200:
        raise TrackingAPIError(response.status_code, response.text[:500], response.text)
    try:
        result = response.json()
    except ValueError as exc:
        raise TrackingAPIError(
            response.status_code,
            "ValidMind returned a non-JSON metric response",
            response.text,
        ) from exc
    if not isinstance(result, dict):
        raise TrackingAPIError(
            response.status_code,
            "ValidMind returned an invalid metric response",
            response.text,
        )
    return result


class MetricsClient:
    """Client for the ValidMind ``log_unit_metric`` endpoint."""

    def __init__(
        self,
        *,
        api_host: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        monitoring: bool = False,
        document: Optional[str] = None,
        client_version: str = "validmind-tracking-core/0.1.0",
        timeout: Optional[float] = None,
        issuer: Optional[str] = None,
        client_id: Optional[str] = None,
        scope: Optional[str] = None,
        audience: Optional[str] = None,
        credentials_path=None,
        status_callback=None,
    ):
        self.api_host = (
            api_url or api_host or os.getenv("VM_API_URL") or os.getenv("VM_API_HOST")
        )
        self.model = model or os.getenv("VM_API_MODEL")
        self.monitoring = monitoring
        self.document = document
        self.client_version = client_version
        self.timeout = float(timeout or os.getenv("VM_API_TIMEOUT", 30))
        self._api_key = api_key if api_key is not None else os.getenv("VM_API_KEY")
        self._api_secret = (
            api_secret if api_secret is not None else os.getenv("VM_API_SECRET")
        )
        issuer = issuer if issuer is not None else os.getenv("VM_OIDC_ISSUER")
        client_id = (
            client_id if client_id is not None else os.getenv("VM_OIDC_CLIENT_ID")
        )
        scope = scope if scope is not None else os.getenv("VM_OIDC_SCOPE")
        audience = audience if audience is not None else os.getenv("VM_OIDC_AUDIENCE")

        has_api_creds = bool(self._api_key and self._api_secret)
        has_oidc = bool(issuer and client_id)
        if not self.api_host:
            raise TrackingConfigurationError("API host must be provided")
        if not self.model:
            raise TrackingConfigurationError("Model ID must be provided")
        if has_api_creds and has_oidc:
            raise TrackingConfigurationError(
                "Provide either API credentials or OIDC credentials, not both"
            )
        if bool(issuer) != bool(client_id):
            raise TrackingConfigurationError(
                "issuer and client_id must be provided together"
            )
        if not has_api_creds and not has_oidc:
            raise TrackingConfigurationError(
                "Provide API credentials or issuer and client_id for OIDC"
            )

        self._oidc = None
        if has_oidc:
            self._oidc = OIDCAuthenticator(
                issuer,
                client_id,
                scope=scope,
                audience=audience,
                timeout=self.timeout,
                credentials_path_value=credentials_path,
                status_callback=status_callback,
            )
            self._oidc.initialize()

    def log_metric(
        self,
        key: str,
        value: Any,
        inputs: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        recorded_at: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        passed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        body = serialize_metric(
            key,
            value,
            inputs,
            params,
            recorded_at,
            thresholds,
            passed,
        )
        return post_metric(
            self._url("log_unit_metric"),
            body,
            self._headers(),
            timeout=self.timeout,
        )

    async def alog_metric(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run the synchronous metric transport without blocking the event loop."""
        return await asyncio.to_thread(self.log_metric, *args, **kwargs)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "X-MODEL-CUID": self.model,
            "X-MONITORING": str(self.monitoring),
            "X-LIBRARY-VERSION": self.client_version,
        }
        if self.document:
            headers["X-DOCUMENT-TYPE"] = self.document
        if self._oidc:
            headers["Authorization"] = f"Bearer {self._oidc.token()}"
        else:
            headers["X-API-KEY"] = self._api_key
            headers["X-API-SECRET"] = self._api_secret
        return headers

    def _url(self, endpoint: str) -> str:
        return urljoin(f"{self.api_host.rstrip('/')}/", endpoint)
