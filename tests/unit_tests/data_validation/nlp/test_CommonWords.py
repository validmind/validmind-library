import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.nlp.CommonWords import CommonWords


class TestCommonWords(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text data
        self.df = pd.DataFrame(
            {
                "text": [
                    "The quick brown fox jumps over the lazy dog",
                    "The fox is quick and brown",
                    "The dog is lazy",
                    "Quick quick quick fox fox",
                    "Brown brown dog",
                ]
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            text_column="text",
            __log=False,
        )

        # Create an invalid dataset without text column
        invalid_df = pd.DataFrame({"numeric": [1, 2, 3, 4, 5]})

        self.invalid_vm_dataset = vm.init_dataset(
            input_id="invalid_dataset",
            dataset=invalid_df,
            __log=False,
        )

    def test_returns_plotly_figure(self):
        # Run the function
        result = CommonWords(self.vm_dataset)

        # Check if result is a Plotly Figure
        self.assertIsInstance(result, go.Figure)

        # Should have one trace (bar chart)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0].type, "bar")

        # Should have a title and axis labels
        self.assertIsNotNone(result.layout.title)
        self.assertIsNotNone(result.layout.xaxis.title)
        self.assertIsNotNone(result.layout.yaxis.title)

    def test_common_words_content(self):
        result = CommonWords(self.vm_dataset)

        # Get the words from the bar chart
        words = result.data[0].x

        # Check that we have some words
        self.assertTrue(len(words) > 0)

        # Check that common words are present
        self.assertIn("fox", words)
        self.assertIn("quick", words)
        self.assertIn("brown", words)
        self.assertIn("dog", words)

        # Check that stopwords are not present
        self.assertNotIn("the", words)
        self.assertNotIn("is", words)
        self.assertNotIn("and", words)
        self.assertNotIn("over", words)

    def test_word_frequencies(self):
        result = CommonWords(self.vm_dataset)

        # Get the words and their frequencies
        words = list(result.data[0].x)
        frequencies = list(result.data[0].y)

        # Find indices for specific words
        fox_idx = words.index("fox")
        quick_idx = words.index("quick")

        # Check frequencies (based on our sample data)
        self.assertEqual(frequencies[fox_idx], 4)  # 'fox' appears 4 times
        self.assertEqual(frequencies[quick_idx], 4)  # 'quick' appears 4 times

    def test_raises_error_for_missing_text_column(self):
        # Check if ValueError is raised when text_column is not specified
        with self.assertRaises(ValueError):
            CommonWords(self.invalid_vm_dataset)
