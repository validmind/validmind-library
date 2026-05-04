# Copyright © 2023-2026 ValidMind Inc. All rights reserved.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validmind.credentials_store import (
    credential_key,
    get_cached_entry,
    is_expired,
    normalize_audience,
    normalize_client_id,
    normalize_issuer,
    upsert_cached_entry,
)
from validmind.errors import ValidMindAuthError


class TestCredentialsStore(unittest.TestCase):
    def test_normalize_issuer_trailing_slash(self):
        self.assertEqual(
            normalize_issuer("https://login.example.com/tenant/v2.0/"),
            "https://login.example.com/tenant/v2.0",
        )

    def test_normalize_issuer_strips_wrapping_quotes(self):
        self.assertEqual(
            normalize_issuer("'https://login.dev.vm.validmind.ai'"),
            "https://login.dev.vm.validmind.ai",
        )
        self.assertEqual(
            normalize_issuer('"https://login.example.com/"'),
            "https://login.example.com",
        )
        self.assertEqual(
            normalize_issuer("\"'https://login.example.com'\""),
            "https://login.example.com",
        )

    def test_normalize_client_id_strips_wrapping_quotes(self):
        self.assertEqual(
            normalize_client_id("'iHpItBjLturosPqnrIFT5S7HbX1IeIUS'"),
            "iHpItBjLturosPqnrIFT5S7HbX1IeIUS",
        )

    def test_credential_key_stable(self):
        k1 = credential_key("https://idp.example.com/", "client-a")
        k2 = credential_key("https://idp.example.com", "client-a")
        self.assertEqual(k1, k2)
        k3 = credential_key("https://idp.example.com", "'client-a'")
        self.assertEqual(k1, k3)

    def test_credential_key_includes_audience(self):
        base = credential_key("https://idp.example.com", "client-a")
        with_aud = credential_key(
            "https://idp.example.com",
            "client-a",
            audience="https://api.example.com",
        )
        self.assertNotEqual(base, with_aud)
        self.assertTrue(with_aud.endswith("|https://api.example.com"))

    def test_normalize_audience(self):
        self.assertEqual(normalize_audience(None), "")
        self.assertEqual(
            normalize_audience("'https://api.example.com'"),
            "https://api.example.com",
        )

    def test_roundtrip_and_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "credentials.json"

            expires_far = "2099-01-01T00:00:00+00:00"
            upsert_cached_entry(
                "https://issuer.example/",
                "cid",
                {
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_at": expires_far,
                },
                path=path,
            )

            entry = get_cached_entry("https://issuer.example", "cid", path=path)
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry["access_token"], "at")
            self.assertFalse(is_expired(entry))

            expired_entry = {**entry, "expires_at": "2020-01-01T00:00:00+00:00"}
            self.assertTrue(is_expired(expired_entry))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "credentials.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValidMindAuthError):
                get_cached_entry("https://x", "y", path=path)

    @patch("validmind.credentials_store.Path.home")
    def test_default_path_under_validmind(self, mock_home):
        mock_home.return_value = Path("/home/tester")
        from validmind.credentials_store import credentials_path

        self.assertEqual(
            credentials_path(), Path("/home/tester/.validmind/credentials.json")
        )
