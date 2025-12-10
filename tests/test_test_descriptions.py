# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import unittest
from unittest.mock import patch

import validmind.ai.test_descriptions as test_desc_module
from validmind.ai.test_descriptions import (
    _estimate_tokens_simple,
    _truncate_summary,
    _truncate_text_simple,
)


class TestTokenEstimation(unittest.TestCase):
    """Test token estimation and truncation functions."""

    def test_estimate_tokens_simple(self):
        """Test simple character-based token estimation."""
        # Test with empty string
        self.assertEqual(_estimate_tokens_simple(""), 0)

        # Test with 100 characters (should be ~25 tokens)
        text_100 = "a" * 100
        self.assertEqual(_estimate_tokens_simple(text_100), 25)

        # Test with 400 characters (should be 100 tokens)
        text_400 = "a" * 400
        self.assertEqual(_estimate_tokens_simple(text_400), 100)

    def test_truncate_text_simple_no_truncation(self):
        """Test that short text is not truncated."""
        short_text = "This is a short text."
        result = _truncate_text_simple(short_text, max_tokens=100)
        self.assertEqual(result, short_text)

    def test_truncate_text_simple_with_truncation(self):
        """Test that long text is truncated correctly."""
        # Create text that's definitely longer than max_tokens
        long_text = "a" * 10000  # 10000 chars = ~2500 tokens

        result = _truncate_text_simple(long_text, max_tokens=100)

        # Should be truncated
        self.assertIn("...[truncated]", result)
        self.assertLess(len(result), len(long_text))

        # Should have beginning and end
        self.assertTrue(result.startswith("a"))
        self.assertTrue(result.endswith("a"))


class TestTruncateSummary(unittest.TestCase):
    """Test the _truncate_summary function."""

    def test_none_and_short_text(self):
        """Test None input and short text that doesn't need truncation."""
        # None input
        self.assertIsNone(_truncate_summary(None, "test.id"))

        # Short text
        short_text = "This is a short summary."
        result = _truncate_summary(short_text, "test.id", max_tokens=100)
        self.assertEqual(result, short_text)

        # Character length optimization (text shorter than max_tokens in chars)
        text = "a" * 50
        result = _truncate_summary(text, "test.id", max_tokens=100)
        self.assertEqual(result, text)

    @patch("validmind.ai.test_descriptions._TIKTOKEN_AVAILABLE", False)
    def test_fallback_truncation(self):
        """Test truncation using fallback when tiktoken is unavailable."""
        long_summary = "y" * 10000  # ~2500 estimated tokens

        result = _truncate_summary(long_summary, "test.id", max_tokens=100)

        # Should be truncated with marker
        self.assertIn("...[truncated]", result)
        self.assertLess(len(result), len(long_summary))
        # Should preserve beginning and end
        self.assertTrue(result.startswith("y"))
        self.assertTrue(result.endswith("y"))


class TestCodePathSelection(unittest.TestCase):
    """Test that the correct code path (tiktoken vs fallback) is selected."""

    def test_module_state(self):
        """Test that module-level flags are set correctly at load time."""
        self.assertIsInstance(test_desc_module._TIKTOKEN_AVAILABLE, bool)

        if test_desc_module._TIKTOKEN_AVAILABLE:
            self.assertIsNotNone(test_desc_module._TIKTOKEN_ENCODING)
        else:
            self.assertIsNone(test_desc_module._TIKTOKEN_ENCODING)

    @patch("validmind.ai.test_descriptions._TIKTOKEN_AVAILABLE", True)
    @patch("validmind.ai.test_descriptions._TIKTOKEN_ENCODING")
    @patch("validmind.ai.test_descriptions._estimate_tokens_simple")
    def test_tiktoken_path(self, mock_estimate, mock_encoding):
        """Test tiktoken path is used when available and fallback is not."""
        mock_encoding.encode.return_value = list(range(1000))
        mock_encoding.decode.return_value = "decoded"

        long_summary = "x" * 10000
        result = _truncate_summary(long_summary, "test.id", max_tokens=100)

        # Verify tiktoken was called
        mock_encoding.encode.assert_called_once_with(long_summary)
        self.assertEqual(mock_encoding.decode.call_count, 2)
        # Verify fallback was NOT called
        mock_estimate.assert_not_called()

        self.assertIn("decoded", result)

    @patch("validmind.ai.test_descriptions._TIKTOKEN_AVAILABLE", False)
    @patch("validmind.ai.test_descriptions._TIKTOKEN_ENCODING")
    @patch("validmind.ai.test_descriptions._estimate_tokens_simple")
    @patch("validmind.ai.test_descriptions._truncate_text_simple")
    def test_fallback_path(self, mock_truncate, mock_estimate, mock_encoding):
        """Test fallback path is used when tiktoken unavailable."""
        mock_estimate.return_value = 1000
        mock_truncate.return_value = "fallback_result"

        long_summary = "y" * 10000
        result = _truncate_summary(long_summary, "test.id", max_tokens=100)

        # Verify fallback was called
        mock_estimate.assert_called_once_with(long_summary)
        mock_truncate.assert_called_once_with(long_summary, 100)
        # Verify tiktoken was NOT called
        mock_encoding.encode.assert_not_called()
        mock_encoding.decode.assert_not_called()

        self.assertEqual(result, "fallback_result")


