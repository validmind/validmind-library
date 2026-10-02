# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests
from validmind_tracking_core import (
    MetricsClient,
    TrackingAuthError,
    TrackingConfigurationError,
    TrackingConnectionError,
    TrackingError,
    post_metric,
    serialize_metric,
)
from validmind_tracking_core.credentials_store import (
    is_expired,
    load_credentials_file,
    upsert_cached_entry,
)
from validmind_tracking_core.oidc import OIDCAuthenticator


class FakeNumpyScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class TestCredentialsStore(unittest.TestCase):
    def test_concurrent_upserts_keep_every_entry(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "credentials.json"
            threads = [
                threading.Thread(
                    target=upsert_cached_entry,
                    args=("https://issuer.example", f"client-{i}", {"a": i}),
                    kwargs={"path": path},
                )
                for i in range(25)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(load_credentials_file(path)["credentials"]), 25)

    def test_is_expired_parses_formats_older_pythons_reject(self):
        for raw in (
            "2999-01-01T00:00:00.123456789+00:00",
            "2999-01-01T00:00:00+0000",
            "2999-01-01T00:00:00.12345Z",
        ):
            self.assertFalse(is_expired({"expires_at": raw}), raw)
        self.assertTrue(is_expired({"expires_at": "2000-01-01T00:00:00+0000"}))


class TestMetricValidation(unittest.TestCase):
    def test_bool_value_rejected(self):
        with self.assertRaisesRegex(ValueError, "passed"):
            serialize_metric("passed", True)

    def test_numpy_style_values_serialize(self):
        body = json.loads(
            serialize_metric("n", FakeNumpyScalar(5), params={"n": FakeNumpyScalar(3)})
        )
        self.assertEqual((body["value"], body["params"]["n"]), (5, 3))

    @patch("validmind_tracking_core.metrics.requests.post")
    def test_network_errors_are_tracking_errors(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("down")
        with self.assertRaises(TrackingConnectionError) as ctx:
            post_metric("https://x.example/log_unit_metric", "{}", {})
        self.assertIsInstance(ctx.exception, TrackingError)


class TestClientConfiguration(unittest.TestCase):
    kwargs = dict(api_host="https://x.example", model="m", api_key="k", api_secret="s")

    def test_blank_timeout_env_is_default(self):
        with patch.dict(os.environ, {"VM_API_TIMEOUT": ""}):
            self.assertEqual(MetricsClient(**self.kwargs).timeout, 30.0)

    def test_invalid_timeout_env_names_variable(self):
        with patch.dict(os.environ, {"VM_API_TIMEOUT": "soon"}):
            with self.assertRaisesRegex(TrackingConfigurationError, "VM_API_TIMEOUT"):
                MetricsClient(**self.kwargs)

    def test_zero_timeout_is_respected(self):
        self.assertEqual(MetricsClient(timeout=0, **self.kwargs).timeout, 0.0)


class TestOIDCSafety(unittest.TestCase):
    def test_http_issuer_rejected_except_loopback(self):
        with self.assertRaises(TrackingConfigurationError):
            OIDCAuthenticator("http://idp.internal", "c")
        with self.assertRaises(TrackingConfigurationError):
            OIDCAuthenticator("idp.internal", "c")
        OIDCAuthenticator("http://localhost:8080", "c")

    @patch("validmind_tracking_core.oidc.run_device_flow")
    def test_non_interactive_by_default(self, mock_flow):
        with TemporaryDirectory() as temp_dir:
            auth = OIDCAuthenticator(
                "https://issuer.example",
                "c",
                credentials_path=Path(temp_dir) / "credentials.json",
            )
            with self.assertRaises(TrackingAuthError):
                auth.initialize()
        mock_flow.assert_not_called()

    @patch("validmind_tracking_core.oidc.requests.get")
    def test_discovery_fetched_once(self, mock_get):
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {
            "device_authorization_endpoint": "https://issuer.example/device",
            "token_endpoint": "https://issuer.example/token",
        }
        auth = OIDCAuthenticator("https://issuer.example", "c")
        auth._token_endpoint()
        auth._token_endpoint()
        self.assertEqual(mock_get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
