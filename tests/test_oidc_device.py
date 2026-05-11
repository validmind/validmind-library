# Copyright © 2023-2026 ValidMind Inc. All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

from validmind import oidc_device as oidc_device_module
from validmind.errors import ValidMindAuthError
from validmind.oidc_device import (
    clear_configuration_cache,
    fetch_openid_configuration,
    poll_device_token,
    refresh_access_token,
    request_device_authorization,
)


class TestOIDCDevice(unittest.TestCase):
    def tearDown(self):
        clear_configuration_cache()

    def test_fetch_openid_configuration_requires_endpoints(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"issuer": "https://example.com"}

        with patch.object(oidc_device_module.requests, "get", return_value=Resp()):
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

    def test_device_authorization_qr_payload_prefers_complete_uri(self):
        payload = oidc_device_module._get_device_authorization_qr_payload(
            "https://example.com/device",
            "https://example.com/device?user_code=ABCD",
        )

        self.assertEqual(payload, "https://example.com/device?user_code=ABCD")

    def test_device_authorization_qr_payload_falls_back_to_verification_uri(self):
        payload = oidc_device_module._get_device_authorization_qr_payload(
            "https://example.com/device",
            None,
        )

        self.assertEqual(payload, "https://example.com/device")

    @patch("builtins.print")
    def test_print_device_authorization_prompt_renders_qr(self, mock_print):
        with patch.object(
            oidc_device_module,
            "_display_device_authorization_qr",
            return_value=True,
        ) as mock_display_qr:
            oidc_device_module._print_device_authorization_prompt(
                "https://example.com/device",
                "ABCD",
                "https://example.com/device?user_code=ABCD",
            )

        mock_display_qr.assert_called_once_with(
            "https://example.com/device",
            "ABCD",
            "https://example.com/device?user_code=ABCD",
        )
        self.assertIn("Visit: https://example.com/device", mock_print.call_args.args[0])

    def test_run_device_flow_passes_complete_uri_to_prompt(self):
        with patch.object(
            oidc_device_module,
            "fetch_openid_configuration",
            return_value={
                "device_authorization_endpoint": "https://example.com/device",
                "token_endpoint": "https://example.com/token",
            },
        ), patch.object(
            oidc_device_module,
            "request_device_authorization",
            return_value={
                "device_code": "dc",
                "user_code": "ABCD",
                "verification_uri": "https://example.com/device",
                "verification_uri_complete": "https://example.com/device?user_code=ABCD",
                "interval": 1,
                "expires_in": 60,
            },
        ), patch.object(
            oidc_device_module,
            "_print_device_authorization_prompt",
        ) as mock_prompt, patch.object(
            oidc_device_module,
            "poll_device_token",
            return_value={"access_token": "tok", "expires_in": 3600},
        ):
            oidc_device_module.run_device_flow(
                "https://issuer.example.com",
                "client-id",
                "openid profile",
            )

        mock_prompt.assert_called_once_with(
            "https://example.com/device",
            "ABCD",
            "https://example.com/device?user_code=ABCD",
        )

    @patch("time.monotonic")
    @patch("time.sleep")
    def test_poll_device_token_success_after_pending(self, mock_sleep, mock_monotonic):
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

        with patch.object(oidc_device_module.requests, "post", return_value=bad):
            with self.assertRaises(ValidMindAuthError):
                refresh_access_token("https://example.com/token", "cid", "bad-refresh")
