# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Dependency-light authentication and tracking primitives for ValidMind SDKs."""

from .errors import (
    TrackingAPIError,
    TrackingAuthError,
    TrackingConfigurationError,
    TrackingConnectionError,
    TrackingError,
)
from .metrics import MetricsClient, post_metric, serialize_metric

__all__ = [
    "MetricsClient",
    "TrackingAPIError",
    "TrackingAuthError",
    "TrackingConfigurationError",
    "TrackingConnectionError",
    "TrackingError",
    "post_metric",
    "serialize_metric",
]
