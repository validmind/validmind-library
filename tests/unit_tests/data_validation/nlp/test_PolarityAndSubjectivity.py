import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.nlp.PolarityAndSubjectivity import (
    PolarityAndSubjectivity,
)


class TestPolarityAndSubjectivity(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text of varying sentiment
        self.df = pd.DataFrame(
            {
                "text": [
                    "I love this product, it is amazing!",  # Positive, subjective
                    "This is terrible, I hate it.",  # Negative, subjective
                    "The sky is blue.",  # Neutral, objective
                    "The product weighs 5 pounds.",  # Neutral, objective
                    "I think this might be good.",  # Slightly positive, subjective
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
        self.invalid_df = pd.DataFrame({"numeric": [1, 2, 3, 4, 5]})

        self.invalid_vm_dataset = vm.init_dataset(
            input_id="invalid_dataset",
            dataset=self.invalid_df,
            __log=False,
        )

    def test_returns_plotly_figure_and_tables_and_raw_data(self):
        # Run the function
        result = PolarityAndSubjectivity(self.vm_dataset)

        # Check if result is a tuple of (Figure, dict, RawData)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        # Check the figure
        fig = result[0]
        self.assertIsInstance(fig, go.Figure)

        # Check if it has scatter trace
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "scatter")

        # Check if it has proper layout
        self.assertIsNotNone(fig.layout.title)
        self.assertIsNotNone(fig.layout.xaxis.title)
        self.assertIsNotNone(fig.layout.yaxis.title)

        # Check the statistics tables
        tables = result[1]
        self.assertIsInstance(tables, dict)
        self.assertIn("Quadrant Distribution", tables)
        self.assertIn("Statistics", tables)

        # Check the raw data
        raw_data = result[2]
        self.assertIsInstance(raw_data, vm.RawData)

    def test_polarity_and_subjectivity_values(self):
        result = PolarityAndSubjectivity(self.vm_dataset)
        stats_df = result[1]["Statistics"]

        # Check polarity range
        polarity_stats = stats_df[stats_df["Metric"] == "Polarity"].iloc[0]
        self.assertEqual(polarity_stats["Range"], "[-1, 1]")
        self.assertTrue(-1 <= polarity_stats["Mean"] <= 1)
        self.assertTrue(-1 <= polarity_stats["Min"] <= 1)
        self.assertTrue(-1 <= polarity_stats["Max"] <= 1)

        # Check subjectivity range
        subjectivity_stats = stats_df[stats_df["Metric"] == "Subjectivity"].iloc[0]
        self.assertEqual(subjectivity_stats["Range"], "[0, 1]")
        self.assertTrue(0 <= subjectivity_stats["Mean"] <= 1)
        self.assertTrue(0 <= subjectivity_stats["Min"] <= 1)
        self.assertTrue(0 <= subjectivity_stats["Max"] <= 1)

    def test_quadrant_distribution(self):
        result = PolarityAndSubjectivity(self.vm_dataset)
        quadrant_df = result[1]["Quadrant Distribution"]

        # Check that all quadrants are present
        expected_quadrants = [
            "Subjective - Positive Sentiment",
            "Subjective - Negative Sentiment",
            "Objective - Positive Sentiment",
            "Objective - Negative Sentiment",
        ]
        self.assertTrue(
            all(q in quadrant_df["Quadrant"].values for q in expected_quadrants)
        )

        # Check that percentages sum to 100%
        total_percentage = quadrant_df["Ratio (%)"].sum()
        self.assertAlmostEqual(total_percentage, 100.0, places=5)

    def test_raises_error_for_missing_text_column(self):
        # Check if ValueError is raised when text_column is not specified
        with self.assertRaises(ValueError):
            PolarityAndSubjectivity(self.invalid_vm_dataset)
