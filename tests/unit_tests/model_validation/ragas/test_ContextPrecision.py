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

from validmind.tests.model_validation.ragas.ContextPrecision import ContextPrecision


class TestContextPrecision(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with questions, contexts, and references
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "What is the capital of France?",
                    "Who wrote Romeo and Juliet?",
                    "What is the largest planet?",
                ],
                "retrieved_contexts": [
                    [
                        "Paris is the capital of France.",  # Most relevant first
                        "France is a country in Europe.",
                        "The Eiffel Tower is in Paris.",
                    ],
                    [
                        "Shakespeare wrote many plays.",  # Less relevant first
                        "Romeo and Juliet was written by William Shakespeare.",  # More relevant
                        "The play was written in the 16th century.",
                    ],
                    [
                        "Mars is the fourth planet from the sun.",  # Irrelevant first
                        "Jupiter is the largest planet in our solar system.",  # Relevant
                        "Saturn is the sixth planet from the sun.",  # Irrelevant
                    ],
                ],
                "reference": [
                    "Paris is the capital city of France.",
                    "William Shakespeare wrote Romeo and Juliet in the 16th century.",
                    "Jupiter is the largest planet in the solar system.",
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="precision_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = ContextPrecision(
            self.vm_dataset,
            user_input_column="user_input",
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

    def test_precision_scores(self):
        """Test if precision scores reflect context ranking quality."""
        result = ContextPrecision(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            reference_column="reference",
        )

        scores = result[0]["Aggregate Scores"][0]

        # Check score ranges
        self.assertTrue(0 <= scores["Mean Score"] <= 1)
        self.assertTrue(0 <= scores["Median Score"] <= 1)
        self.assertTrue(0 <= scores["Max Score"] <= 1)
        self.assertTrue(0 <= scores["Min Score"] <= 1)
        self.assertTrue(scores["Standard Deviation"] >= 0)
        self.assertEqual(scores["Count"], len(self.df))

        # First example has relevant context first, should contribute to higher scores
        self.assertGreater(
            scores["Max Score"], 0.7, "Well-ranked contexts should have high precision"
        )

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            ContextPrecision(
                self.vm_dataset,
                user_input_column="invalid_column",
                retrieved_contexts_column="retrieved_contexts",
                reference_column="reference",
            )
