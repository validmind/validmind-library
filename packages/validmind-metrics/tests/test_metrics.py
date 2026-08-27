# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import asyncio
import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from validmind_metrics import MetricsClient
from validmind_tracking_core.credentials_store import upsert_cached_entry


class TestMetricsClient(unittest.TestCase):
    def _response(self, body=None, status_code=200, text=""):
        response = Mock()
        response.status_code = status_code
        response.text = text
        response.json.return_value = body or {"ok": True}
        return response

    @patch("validmind_tracking_core.metrics.requests.post")
    def test_log_metric_uses_api_key_headers_and_schema(self, mock_post):
        mock_post.return_value = self._response({"metric_id": "metric-1"})
        client = MetricsClient(
            api_host="https://tracking.example/api/v1/tracking",
            model="model-1",
            api_key="key",
            api_secret="secret",
            monitoring=True,
            document="monitoring",
            client_version="test-client/1.0",
        )

        result = client.log_metric(
            "accuracy",
            0.95,
            inputs=["dataset-1"],
            params={"average": "macro"},
            recorded_at="2026-08-27T00:00:00Z",
            thresholds={"minimum": 0.9},
            passed=True,
        )

        self.assertEqual(result, {"metric_id": "metric-1"})
        mock_post.assert_called_once()
        url = mock_post.call_args.args[0]
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(
            url, "https://tracking.example/api/v1/tracking/log_unit_metric"
        )
        self.assertEqual(
            kwargs["headers"],
            {
                "X-MODEL-CUID": "model-1",
                "X-MONITORING": "True",
                "X-LIBRARY-VERSION": "test-client/1.0",
                "X-DOCUMENT-TYPE": "monitoring",
                "X-API-KEY": "key",
                "X-API-SECRET": "secret",
            },
        )
        self.assertEqual(
            json.loads(kwargs["data"]),
            {
                "key": "accuracy",
                "value": 0.95,
                "inputs": ["dataset-1"],
                "params": {"average": "macro"},
                "recorded_at": "2026-08-27T00:00:00Z",
                "thresholds": {"minimum": 0.9},
                "passed": True,
            },
        )

    @patch("validmind_tracking_core.metrics.requests.post")
    def test_cached_oidc_token_is_used(self, mock_post):
        mock_post.return_value = self._response({"metric_id": "metric-2"})
        with TemporaryDirectory() as temp_dir:
            credentials_path = Path(temp_dir) / "credentials.json"
            upsert_cached_entry(
                "https://issuer.example",
                "client-1",
                {
                    "access_token": "cached-token",
                    "refresh_token": "refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).isoformat(),
                },
                path=credentials_path,
            )

            client = MetricsClient(
                api_host="https://tracking.example/api/v1/tracking",
                model="model-1",
                issuer="https://issuer.example/",
                client_id="client-1",
                credentials_path=credentials_path,
            )
            client.log_metric("accuracy", 0.95)

        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer cached-token")
        self.assertNotIn("X-API-KEY", headers)

    @patch("validmind_tracking_core.metrics.requests.post")
    def test_async_metric_does_not_require_nested_event_loop(self, mock_post):
        mock_post.return_value = self._response({"ok": True})
        client = MetricsClient(
            api_host="https://tracking.example/api/v1/tracking",
            model="model-1",
            api_key="key",
            api_secret="secret",
        )

        async def handler():
            return await client.alog_metric("accuracy", 0.95)

        self.assertEqual(asyncio.run(handler()), {"ok": True})

    def test_import_isolated_from_full_library(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import validmind_metrics; "
                "assert 'validmind' not in sys.modules; "
                "assert 'aiohttp' not in sys.modules",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
