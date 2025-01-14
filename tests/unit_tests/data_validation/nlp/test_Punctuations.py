import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.nlp.Punctuations import Punctuations, RawData


class TestPunctuations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text containing various punctuation
        self.df = pd.DataFrame(
            {
                "text": [
                    "Hello, world!",  # comma and exclamation
                    "Is this a test? Yes, it is.",  # question mark, comma, period
                    "This... is interesting.",  # ellipsis, period
                    'The "quick" brown fox.',  # quotes, period
                    "Semi-colons; are fun!",  # hyphen, semicolon, exclamation
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
        # Run the function with default token mode
        fig, raw_data = Punctuations(self.vm_dataset)

        # Check if result is a Plotly Figure
        self.assertIsInstance(fig, go.Figure)

        # Should have one trace (bar chart)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "bar")

        # Should have a title and axis labels
        self.assertIsNotNone(fig.layout.title)
        self.assertIsNotNone(fig.layout.xaxis.title)
        self.assertIsNotNone(fig.layout.yaxis.title)

        # Check that raw_data is instance of RawData
        self.assertIsInstance(raw_data, RawData)

    def test_token_mode_counting(self):
        fig, raw_data = Punctuations(self.vm_dataset, count_mode="token")

        # Get the punctuation marks and their counts
        punctuation_marks = fig.data[0].x
        counts = fig.data[0].y

        # Convert to dict for easier testing
        punct_counts = dict(zip(punctuation_marks, counts))

        # Check specific punctuation counts
        self.assertEqual(punct_counts[","], 0)  # Zero commas in the test data
        self.assertEqual(punct_counts["."], 0)  # Zero periods
        self.assertEqual(punct_counts["!"], 0)  # Zero exclamation marks
        self.assertEqual(punct_counts["?"], 0)  # Zero question marks

        # Check that raw_data is instance of RawData
        self.assertIsInstance(raw_data, RawData)

    def test_word_mode_counting(self):
        fig, raw_data = Punctuations(self.vm_dataset, count_mode="word")

        # Get the punctuation marks and their counts
        punctuation_marks = fig.data[0].x
        counts = fig.data[0].y

        # Convert to dict for easier testing
        punct_counts = dict(zip(punctuation_marks, counts))

        # Check specific punctuation counts (should include punctuation within words)
        self.assertTrue(punct_counts["-"] > 0)  # Should count hyphen in "Semi-colons"
        self.assertEqual(punct_counts['"'], 2)  # Two quote marks

        # Check that raw_data is instance of RawData
        self.assertIsInstance(raw_data, RawData)

    def test_invalid_count_mode(self):
        # Check if ValueError is raised for invalid count_mode
        with self.assertRaises(ValueError):
            Punctuations(self.vm_dataset, count_mode="invalid")

    def test_raises_error_for_missing_text_column(self):
        # Check if ValueError is raised when text_column is not specified
        with self.assertRaises(ValueError):
            Punctuations(self.invalid_vm_dataset)
