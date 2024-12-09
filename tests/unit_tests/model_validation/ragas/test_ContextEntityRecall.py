import os
import dotenv
import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm

# Load environment variables at the start of the test file
dotenv.load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is not set")

from validmind.tests.model_validation.ragas.ContextEntityRecall import (
    ContextEntityRecall,
)


class TestContextEntityRecall(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with references and contexts containing entities
        self.df = pd.DataFrame(
            {
                "reference": [
                    "The Eiffel Tower in Paris, France was completed in 1889.",
                    "Albert Einstein developed the theory of relativity in 1915.",
                    "The Great Wall of China was built during the Ming Dynasty.",
                    "William Shakespeare wrote Romeo and Juliet in the 16th century.",
                ],
                "retrieved_contexts": [
                    [
                        "The Eiffel Tower stands in Paris. It was built for the 1889 World's Fair."
                    ],
                    [
                        "Einstein's theory of relativity revolutionized physics in the early 20th century."
                    ],
                    ["The Ming Dynasty oversaw major construction of the Great Wall."],
                    [
                        "Shakespeare's famous play Romeo and Juliet was written between 1591-1595."
                    ],
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="context_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = ContextEntityRecall(
            self.vm_dataset,
            retrieved_contexts_column="retrieved_contexts",
            reference_column="reference",
        )

        # Check return types
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)  # dict and 2 figures

        # Check dictionary structure
        self.assertIsInstance(result[0], dict)
        self.assertIn("Aggregate Scores", result[0])
        self.assertIsInstance(result[0]["Aggregate Scores"], list)
        self.assertEqual(len(result[0]["Aggregate Scores"]), 1)

        # Check figures
        self.assertIsInstance(result[1], go.Figure)  # Histogram
        self.assertIsInstance(result[2], go.Figure)  # Box plot

    def test_aggregate_scores(self):
        """Test if aggregate scores have expected structure and values."""
        result = ContextEntityRecall(
            self.vm_dataset,
            retrieved_contexts_column="retrieved_contexts",
            reference_column="reference",
        )

        scores = result[0]["Aggregate Scores"][0]

        # Check score keys
        expected_keys = [
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
            "Count",
        ]
        self.assertListEqual(list(scores.keys()), expected_keys)

        # Check score ranges
        self.assertTrue(0 <= scores["Mean Score"] <= 1)
        self.assertTrue(0 <= scores["Median Score"] <= 1)
        self.assertTrue(0 <= scores["Max Score"] <= 1)
        self.assertTrue(0 <= scores["Min Score"] <= 1)
        self.assertTrue(scores["Standard Deviation"] >= 0)
        self.assertEqual(scores["Count"], len(self.df))

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            ContextEntityRecall(
                self.vm_dataset,
                retrieved_contexts_column="invalid_column",
                reference_column="reference",
            )

    def test_matching_entities(self):
        """Test with contexts containing exact entity matches."""
        # Create dataset with perfect entity matches
        df_perfect = pd.DataFrame(
            {
                "reference": ["The Eiffel Tower is in Paris."],
                "retrieved_contexts": [["The Eiffel Tower stands in Paris."]],
            }
        )

        vm_dataset_perfect = vm.init_dataset(
            input_id="perfect_dataset",
            dataset=df_perfect,
            __log=False,
        )

        result = ContextEntityRecall(
            vm_dataset_perfect,
            retrieved_contexts_column="retrieved_contexts",
            reference_column="reference",
        )

        scores = result[0]["Aggregate Scores"][0]
        # Score should be high for matching entities
        self.assertGreater(scores["Mean Score"], 0.8)
