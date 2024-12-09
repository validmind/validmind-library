import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.nlp.LanguageDetection import LanguageDetection


class TestLanguageDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text in different languages
        self.df = pd.DataFrame(
            {
                "text": [
                    "Hello world, this is English text",  # English
                    "Bonjour le monde, ceci est du texte français",  # French
                    "Hola mundo, este es texto en español",  # Spanish
                    "Hallo Welt, dies ist deutscher Text",  # German
                    "123 456",  # Should return "Unknown"
                ]
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            text_column="text",  # Specify the text column
            __log=False,
        )

        # Create an invalid dataset without text column
        self.invalid_df = pd.DataFrame({"numeric": [1, 2, 3, 4, 5]})

        self.invalid_vm_dataset = vm.init_dataset(
            input_id="invalid_dataset",
            dataset=self.invalid_df,
            __log=False,
        )

    def test_returns_plotly_figure(self):
        # Run the function
        result = LanguageDetection(self.vm_dataset)

        # Check if result is a Plotly Figure
        self.assertIsInstance(result, go.Figure)

        # Should have one trace (histogram)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0].type, "histogram")

        # Should have a title and axis labels
        self.assertIsNotNone(result.layout.title)
        self.assertIsNotNone(result.layout.xaxis.title)

    def test_language_detection(self):
        result = LanguageDetection(self.vm_dataset)

        # Get the detected languages from the histogram
        languages = result.data[0].x

        # Check that expected languages are present
        self.assertTrue("en" in languages[0])  # English
        self.assertTrue("fr" in languages[1])  # French
        self.assertTrue("es" in languages[2])  # Spanish
        self.assertTrue("de" in languages[3])  # German
        self.assertTrue("Unknown" in languages[4])  # For numeric text

    def test_raises_error_for_missing_text_column(self):
        # Check if ValueError is raised when text_column is not specified
        with self.assertRaises(ValueError):
            LanguageDetection(self.invalid_vm_dataset)
