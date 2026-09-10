# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Lightweight metric logging client for the ValidMind Platform."""

from typing import Any, Dict, List, Optional

from validmind_tracking_core import (
    MetricsClient,
    TrackingAPIError,
    TrackingAuthError,
    TrackingConfigurationError,
)

__version__ = "0.1.0"

_client: Optional[MetricsClient] = None


def init(**kwargs: Any) -> MetricsClient:
    """Create and retain the default metric client."""
    global _client
    kwargs.setdefault("client_version", __version__)
    _client = MetricsClient(**kwargs)
    return _client


def _get_client() -> MetricsClient:
    if _client is None:
        return init()
    return _client


def log_metric(
    key: str,
    value: Any,
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    passed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Log one metric using the default client."""
    return _get_client().log_metric(
        key,
        value,
        inputs=inputs,
        params=params,
        recorded_at=recorded_at,
        thresholds=thresholds,
        passed=passed,
    )


async def alog_metric(
    key: str,
    value: Any,
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    passed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Log one metric without blocking the current event loop."""
    return await _get_client().alog_metric(
        key,
        value,
        inputs=inputs,
        params=params,
        recorded_at=recorded_at,
        thresholds=thresholds,
        passed=passed,
    )


__all__ = [
    "MetricsClient",
    "TrackingAPIError",
    "TrackingAuthError",
    "TrackingConfigurationError",
    "alog_metric",
    "init",
    "log_metric",
]
