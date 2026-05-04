# Copyright © 2023-2026 ValidMind Inc. All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

from validmind.errors import ValidMindAuthError
from validmind.oidc_device import (
    clear_configuration_cache,
    fetch_openid_configuration,
    poll_device_token,
    refresh_access_token,
    request_device_authorization,
)
from validmind import oidc_device as oidc_device_module


class TestOIDCDevice(unittest.TestCase):
    def tearDown(self):
        clear_configuration_cache()

    def test_fetch_openid_configuration_requires_endpoints(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"issuer": "https://example.com"}

        with patch.object(
            oidc_device_module.requests, "get", return_value=Resp()
        ):
            with self.assertRaises(ValidMindAuthError) as ctx:
                fetch_openid_configuration("https://example.com")
            self.assertIn("device_authorization_endpoint", str(ctx.exception))

    def test_request_device_authorization_success(self):
        class Resp:
            status_code = 200

            def json(self):
                return {
                    "device_code": "dc",
                    "user_code": "ABCD",
                    "verification_uri": "https://example.com/device",
                    "interval": 1,
                    "expires_in": 60,
                }

        with patch.object(
            oidc_device_module.requests, "post", return_value=Resp()
        ) as mock_post:
            out = request_device_authorization(
                "https://example.com/device", "client", "openid profile"
            )
        self.assertEqual(out["device_code"], "dc")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("client_id", kwargs["data"])
        self.assertNotIn("audience", kwargs["data"])

    def test_request_device_authorization_includes_audience(self):
        class Resp:
            status_code = 200

            def json(self):
                return {
                    "device_code": "dc",
                    "user_code": "ABCD",
                    "verification_uri": "https://example.com/device",
                    "interval": 1,
                    "expires_in": 60,
                }

        with patch.object(
            oidc_device_module.requests, "post", return_value=Resp()
        ) as mock_post:
            request_device_authorization(
                "https://example.com/device",
                "client",
                "openid profile",
                audience="https://api.example.com",
            )
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["data"]["audience"], "https://api.example.com")

    @patch("time.monotonic")
    @patch("time.sleep")
    def test_poll_device_token_success_after_pending(
        self, mock_sleep, mock_monotonic
    ):
        pending = MagicMock()
        pending.status_code = 401
        pending.json.return_value = {"error": "authorization_pending"}

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {
            "access_token": "token",
            "expires_in": 3600,
            "refresh_token": "rt",
        }

        mock_monotonic.side_effect = [0, 10, 20, 100]
        with patch.object(
            oidc_device_module.requests,
            "post",
            side_effect=[pending, success],
        ):
            tok = poll_device_token(
                "https://example.com/token",
                "client-id",
                "device-code",
                interval=1,
                expires_in=50,
            )
        self.assertEqual(tok["access_token"], "token")
        self.assertEqual(mock_sleep.call_count, 1)

    def test_refresh_access_token_failure(self):
        bad = MagicMock()
        bad.status_code = 400
        bad.json.return_value = {"error": "invalid_grant"}
        bad.text = ""

        with patch.object(
            oidc_device_module.requests, "post", return_value=bad
        ):
            with self.assertRaises(ValidMindAuthError):
                refresh_access_token(
                    "https://example.com/token", "cid", "bad-refresh"
                )
