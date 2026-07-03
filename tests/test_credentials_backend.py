# Copyright © 2023-2026 ValidMind Inc. All rights reserved.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validmind.credentials_backend import (
    FileCredentialsBackend,
    MemoryCredentialsBackend,
    NullCredentialsBackend,
    load_backend_from_env,
    resolve_credentials_backend,
    set_default_backend,
)
from validmind.credentials_store import credential_key, upsert_cached_entry
from validmind.errors import ValidMindAuthError


class TestMemoryCredentialsBackend(unittest.TestCase):
    def test_roundtrip(self):
        backend = MemoryCredentialsBackend()
        key = credential_key("https://issuer/", "cid")
        row = {"access_token": "at", "expires_at": "2099-01-01T00:00:00+00:00"}
        backend.put(key, row)
        self.assertEqual(backend.get(key), row)
        backend.delete(key)
        self.assertIsNone(backend.get(key))

    def test_null_alias(self):
        self.assertIs(NullCredentialsBackend, MemoryCredentialsBackend)


class TestFileCredentialsBackend(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "credentials.json"
            backend = FileCredentialsBackend(path=path)
            key = credential_key("https://issuer/", "cid")
            row = {"access_token": "at", "expires_at": "2099-01-01T00:00:00+00:00"}
            backend.put(key, row)
            self.assertEqual(backend.get(key), row)
            backend.delete(key)
            self.assertIsNone(backend.get(key))


class TestResolveCredentialsBackend(unittest.TestCase):
    def tearDown(self):
        set_default_backend(None)

    def test_explicit_wins_over_default(self):
        explicit = MemoryCredentialsBackend()
        default = MemoryCredentialsBackend()
        set_default_backend(default)
        self.assertIs(resolve_credentials_backend(explicit), explicit)

    def test_default_backend(self):
        backend = MemoryCredentialsBackend()
        set_default_backend(backend)
        self.assertIs(resolve_credentials_backend(), backend)

    @patch.dict("os.environ", {"VM_OIDC_NO_PERSIST": "1"}, clear=True)
    def test_env_no_persist(self):
        backend = resolve_credentials_backend()
        self.assertIsInstance(backend, MemoryCredentialsBackend)

    @patch.dict(
        "os.environ",
        {
            "VM_OIDC_CREDENTIALS_BACKEND": (
                "validmind.credentials_backend:MemoryCredentialsBackend"
            )
        },
        clear=True,
    )
    def test_env_backend_spec(self):
        backend = resolve_credentials_backend()
        self.assertIsInstance(backend, MemoryCredentialsBackend)

    @patch.dict(
        "os.environ",
        {"VM_OIDC_CREDENTIALS_BACKEND": "not-a-valid-spec"},
        clear=True,
    )
    def test_invalid_env_spec_raises(self):
        with self.assertRaises(ValidMindAuthError):
            resolve_credentials_backend()

    @patch.dict("os.environ", {}, clear=True)
    def test_default_is_file_backend(self):
        backend = resolve_credentials_backend()
        self.assertIsInstance(backend, FileCredentialsBackend)


class TestLoadBackendFromEnv(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_unset(self):
        self.assertIsNone(load_backend_from_env())

    @patch.dict("os.environ", {"VM_OIDC_NO_PERSIST": "true"}, clear=True)
    def test_no_persist(self):
        self.assertIsInstance(load_backend_from_env(), MemoryCredentialsBackend)


class TestCredentialsStoreWithBackend(unittest.TestCase):
    def test_upsert_uses_explicit_backend(self):
        backend = MemoryCredentialsBackend()
        upsert_cached_entry(
            "https://issuer/",
            "cid",
            {"access_token": "tok", "expires_at": "2099-01-01T00:00:00+00:00"},
            backend=backend,
        )
        key = credential_key("https://issuer/", "cid")
        self.assertEqual(backend.get(key)["access_token"], "tok")
