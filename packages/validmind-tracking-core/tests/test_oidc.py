# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root directory for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from validmind_tracking_core.credentials_store import (
    get_cached_entry,
    upsert_cached_entry,
)
from validmind_tracking_core.oidc import OIDCAuthenticator, _bearer_token


class TestOIDCAuthenticator(unittest.TestCase):
    def test_bearer_token_matches_entra_hostname(self):
        self.assertEqual(
            _bearer_token(
                {
                    "issuer": "https://login.microsoftonline.com/tenant/v2.0",
                    "access_token": "access-token",
                    "id_token": "id-token",
                }
            ),
            "id-token",
        )
        self.assertEqual(
            _bearer_token(
                {
                    "issuer": "https://login.microsoftonline.com.evil.example/tenant",
                    "access_token": "access-token",
                    "id_token": "id-token",
                }
            ),
            "access-token",
        )

    @patch("validmind_tracking_core.oidc.requests.post")
    @patch("validmind_tracking_core.oidc.requests.get")
    def test_refreshes_expired_cached_token(self, mock_get, mock_post):
        discovery = Mock(status_code=200)
        discovery.json.return_value = {
            "device_authorization_endpoint": "https://issuer.example/device",
            "token_endpoint": "https://issuer.example/token",
        }
        mock_get.return_value = discovery
        refreshed = Mock(status_code=200)
        refreshed.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 3600,
        }
        mock_post.return_value = refreshed

        with TemporaryDirectory() as temp_dir:
            credentials_path = Path(temp_dir) / "credentials.json"
            upsert_cached_entry(
                "https://issuer.example",
                "client-1",
                {
                    "access_token": "expired-token",
                    "refresh_token": "refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(hours=1)
                    ).isoformat(),
                },
                path=credentials_path,
            )
            auth = OIDCAuthenticator(
                "https://issuer.example",
                "client-1",
                credentials_path_value=credentials_path,
            )
            auth.initialize()

            self.assertEqual(auth.token(), "refreshed-token")
            self.assertEqual(
                mock_post.call_args.args[0], "https://issuer.example/token"
            )
            self.assertEqual(
                get_cached_entry(
                    "https://issuer.example", "client-1", path=credentials_path
                )["refresh_token"],
                "refresh-token",
            )


if __name__ == "__main__":
    unittest.main()
