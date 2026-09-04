# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Errors raised by the dependency-light tracking core."""


class TrackingError(Exception):
    """Base class for tracking-core failures."""


class TrackingConfigurationError(TrackingError):
    """The tracking client configuration is invalid or incomplete."""


class TrackingAuthError(TrackingError):
    """API-key or OIDC authentication failed."""


class TrackingAPIError(TrackingError):
    """The tracking API rejected a request or returned an invalid response."""

    def __init__(self, status_code: int, message: str, response_text: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
