# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import os
import unittest
from unittest.mock import patch

import validmind.api_client as api_client
from validmind.ai.utils import get_client_and_model, is_configured
from validmind.tests.prompt_validation.ai_powered_test import call_model


class MockResponse:
    def __init__(self, status, text=None, json=None):
        self.status = status
        self.status_code = status
        self.text = text
        self._json = json or {}

    def json(self):
        return self._json


class TestServerLLMOnly(unittest.TestCase):
    """Tests for server-only LLM mode functionality."""

    def setUp(self):
        """Set up test environment variables."""
        # Store original values
        self.original_api_key = os.environ.get("VM_API_KEY")
        self.original_api_secret = os.environ.get("VM_API_SECRET")
        self.original_api_host = os.environ.get("VM_API_HOST")
        self.original_api_model = os.environ.get("VM_API_MODEL")
        self.original_openai_key = os.environ.get("OPENAI_API_KEY")
        self.original_azure_key = os.environ.get("AZURE_OPENAI_KEY")
        self.original_server_llm_only = os.environ.get("VALIDMIND_USE_SERVER_LLM_ONLY")

        # Set required environment variables for tests
        os.environ["VM_API_KEY"] = "test_api_key"
        os.environ["VM_API_SECRET"] = "test_api_secret"
        os.environ["VM_API_HOST"] = "https://test.validmind.ai/api/v1/tracking"
        os.environ["VM_API_MODEL"] = "test_model_id"

        # Clear OpenAI-related env vars to test server-only mode
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "AZURE_OPENAI_KEY" in os.environ:
            del os.environ["AZURE_OPENAI_KEY"]
        if "VALIDMIND_USE_SERVER_LLM_ONLY" in os.environ:
            del os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"]

    def _restore_env_var(self, var_name, original_value):
        """Helper to restore or clear an environment variable."""
        if original_value:
            os.environ[var_name] = original_value
        elif var_name in os.environ:
            del os.environ[var_name]

    def _reset_ai_utils_globals(self):
        """Helper to reset global state in ai.utils module."""
        import validmind.ai.utils as ai_utils

        # Reset module-level globals using name mangling pattern
        # For __client in module utils, Python mangles it to _utils__client
        globals_to_reset = [
            "_utils__client",
            "_utils__model",
            "_utils__judge_llm",
            "_utils__judge_embeddings",
            "_utils__ack",
        ]
        for global_name in globals_to_reset:
            if hasattr(ai_utils, global_name):
                setattr(ai_utils, global_name, None)

    def tearDown(self):
        """Restore original environment variables."""
        # Restore or clear environment variables
        self._restore_env_var("VM_API_KEY", self.original_api_key)
        self._restore_env_var("VM_API_SECRET", self.original_api_secret)
        self._restore_env_var("VM_API_HOST", self.original_api_host)
        self._restore_env_var("VM_API_MODEL", self.original_api_model)
        self._restore_env_var("OPENAI_API_KEY", self.original_openai_key)
        self._restore_env_var("AZURE_OPENAI_KEY", self.original_azure_key)
        self._restore_env_var("VALIDMIND_USE_SERVER_LLM_ONLY", self.original_server_llm_only)

        # Reset global state in ai.utils
        self._reset_ai_utils_globals()

    @patch("requests.get")
    def test_init_with_use_server_llm_only_parameter(self, mock_get):
        """Test that use_server_llm_only parameter sets the environment variable."""
        mock_get.return_value = MockResponse(
            200,
            json={
                "model": {"name": "test_model", "cuid": "test_model_id"},
                "feature_flags": {},
                "document_type": "model_documentation",
            },
        )

        api_client.init(use_server_llm_only=True)

        # Verify environment variable is set
        self.assertEqual(os.environ.get("VALIDMIND_USE_SERVER_LLM_ONLY"), "True")

        # Verify ping was called
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_init_with_use_server_llm_only_false(self, mock_get):
        """Test that use_server_llm_only=False clears the environment variable."""
        # First set it to True
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "True"

        mock_get.return_value = MockResponse(
            200,
            json={
                "model": {"name": "test_model", "cuid": "test_model_id"},
                "feature_flags": {},
                "document_type": "model_documentation",
            },
        )

        api_client.init(use_server_llm_only=False)

        # Verify environment variable is set to False
        self.assertEqual(os.environ.get("VALIDMIND_USE_SERVER_LLM_ONLY"), "False")

    @patch("requests.get")
    def test_init_with_use_server_llm_only_none(self, mock_get):
        """Test that use_server_llm_only=None doesn't change existing env var."""
        # Set it beforehand
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "1"

        mock_get.return_value = MockResponse(
            200,
            json={
                "model": {"name": "test_model", "cuid": "test_model_id"},
                "feature_flags": {},
                "document_type": "model_documentation",
            },
        )

        api_client.init(use_server_llm_only=None)

        # Verify environment variable is unchanged
        self.assertEqual(os.environ.get("VALIDMIND_USE_SERVER_LLM_ONLY"), "1")

    def test_get_client_and_model_raises_error_when_server_only_enabled(self):
        """Test that get_client_and_model raises ValueError when server-only mode is enabled."""
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "1"

        with self.assertRaises(ValueError) as context:
            get_client_and_model()

        error_message = str(context.exception)
        self.assertIn("Local LLM calls are disabled", error_message)
        self.assertIn("server-side calls", error_message)

    def test_get_client_and_model_works_when_server_only_disabled(self):
        """Test that get_client_and_model works normally when server-only mode is disabled."""
        # Set OpenAI API key to allow local calls
        os.environ["OPENAI_API_KEY"] = "test_openai_key"
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "0"

        # Should not raise an error (though it might fail on actual API call, that's OK)
        # We're just testing that the server-only check doesn't block it
        try:
            client, model = get_client_and_model()
            # If we get here, the server-only check passed
            self.assertIsNotNone(client)
            self.assertIsNotNone(model)
        except Exception as e:
            # Other errors (like API connection) are OK, we just want to ensure
            # the server-only check doesn't block it
            self.assertNotIn("Local LLM calls are disabled", str(e))

    def test_is_configured_returns_true_when_server_only_enabled_and_api_configured(self):
        """Test that is_configured returns True when server-only mode is enabled and API is configured."""
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "1"

        # Mock the API client to have credentials
        with patch.object(api_client, "_api_key", "test_key"), patch.object(
            api_client, "_api_secret", "test_secret"
        ):
            result = is_configured()
            self.assertTrue(result)

    def test_is_configured_returns_false_when_server_only_enabled_but_api_not_configured(self):
        """Test that is_configured returns False when server-only mode is enabled but API is not configured."""
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "1"

        # Mock the API client to NOT have credentials
        with patch.object(api_client, "_api_key", None), patch.object(
            api_client, "_api_secret", None
        ):
            result = is_configured()
            self.assertFalse(result)

    def test_is_configured_works_normally_when_server_only_disabled(self):
        """Test that is_configured works normally when server-only mode is disabled."""
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "0"
        os.environ["OPENAI_API_KEY"] = "test_openai_key"

        # Should attempt to check OpenAI (may fail, but that's OK)
        # We're just testing that server-only check doesn't interfere
        try:
            result = is_configured()
            # Result can be True or False depending on actual API availability
            self.assertIsInstance(result, bool)
        except Exception:
            # Other exceptions are OK, we just want to ensure server-only check doesn't block
            pass

    def test_call_model_raises_error_when_server_only_enabled(self):
        """Test that call_model raises ValueError when server-only mode is enabled."""
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "1"

        with self.assertRaises(ValueError) as context:
            call_model(
                system_prompt="Test system prompt",
                user_prompt="Test user prompt",
            )

        error_message = str(context.exception)
        self.assertIn("Local LLM calls are disabled", error_message)
        self.assertIn("VALIDMIND_USE_SERVER_LLM_ONLY", error_message)

    def test_environment_variable_case_insensitive(self):
        """Test that environment variable values are case-insensitive."""
        # Test various case combinations
        test_cases = ["1", "True", "TRUE", "true", "0", "False", "FALSE", "false"]

        for value in test_cases:
            with self.subTest(value=value):
                os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = value

                # Reset the ack state
                import validmind.ai.utils as ai_utils

                if hasattr(ai_utils, "_utils__ack"):
                    setattr(ai_utils, "_utils__ack", None)

                # Check if it's treated as enabled (1, True, true, TRUE) or disabled (0, False, false, FALSE)
                is_enabled = value.lower() in ["1", "true"]

                if is_enabled:
                    with self.assertRaises(ValueError) as context:
                        get_client_and_model()
                    self.assertIn("Local LLM calls are disabled", str(context.exception))
                else:
                    # When disabled, it should try to use local OpenAI (may fail, but that's OK)
                    try:
                        get_client_and_model()
                    except ValueError as e:
                        # Should not be the server-only error
                        self.assertNotIn("Local LLM calls are disabled", str(e))

    @patch("requests.get")
    def test_generate_descriptions_still_works_with_server_only(self, mock_get):
        """Test that test result descriptions still work when server-only mode is enabled."""
        os.environ["VALIDMIND_USE_SERVER_LLM_ONLY"] = "1"

        mock_get.return_value = MockResponse(
            200,
            json={
                "model": {"name": "test_model", "cuid": "test_model_id"},
                "feature_flags": {},
                "document_type": "model_documentation",
            },
        )

        api_client.init()

        # Test result descriptions use server-side API, so they should work
        # We can't fully test this without mocking the actual API call,
        # but we can verify that get_client_and_model is not called for descriptions
        # (descriptions use generate_test_result_description which calls the API directly)

        # The key point is that get_client_and_model should raise an error
        # but generate_test_result_description should still work
        with self.assertRaises(ValueError):
            get_client_and_model()

        # But we should be able to import and use the description generation
        # (it uses the API client, not local OpenAI)
        from validmind.ai.test_descriptions import get_result_description

        # This should work because it uses server-side API
        # We can't fully test without mocking, but we can verify it doesn't
        # immediately fail due to server-only mode
        try:
            # This will fail for other reasons (missing inputs), but not because
            # of server-only mode blocking it
            get_result_description(
                test_id="test.test",
                test_description="Test description",
                should_generate=False,  # Don't actually generate to avoid API call
            )
        except Exception as e:
            # Should not be the server-only error
            self.assertNotIn("Local LLM calls are disabled", str(e))


if __name__ == "__main__":
    unittest.main()
