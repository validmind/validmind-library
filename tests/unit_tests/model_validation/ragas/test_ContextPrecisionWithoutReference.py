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

from validmind.tests.model_validation.ragas.ContextPrecisionWithoutReference import (
    ContextPrecisionWithoutReference,
)


class TestContextPrecisionWithoutReference(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with questions, contexts, and responses
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "What is the capital of France?",
                    "Who wrote Romeo and Juliet?",
                    "What is the largest planet?",
                ],
                "retrieved_contexts": [
                    [
                        "Paris is the capital of France.",  # Highly relevant
                        "France is a country in Europe.",  # Somewhat relevant
                        "The Eiffel Tower is in Paris.",  # Less relevant
                    ],
                    [
                        "Romeo and Juliet was written by William Shakespeare.",  # Highly relevant
                        "Shakespeare wrote many famous plays.",  # Somewhat relevant
                        "The play was written in the 16th century.",  # Less relevant
                    ],
                    [
                        "Mars is the fourth planet from the sun.",  # Not relevant
                        "Jupiter is the largest planet in our solar system.",  # Highly relevant
                        "Saturn has beautiful rings.",  # Not relevant
                    ],
                ],
                "response": [
                    "The capital of France is Paris.",
                    "William Shakespeare wrote Romeo and Juliet.",
                    "Jupiter is the largest planet in our solar system.",
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
        result = ContextPrecisionWithoutReference(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
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
        """Test if precision scores reflect context relevance."""
        result = ContextPrecisionWithoutReference(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
        )

        scores = result[0]["Aggregate Scores"][0]

        # Check score ranges
        self.assertTrue(0 <= scores["Mean Score"] <= 1)
        self.assertTrue(0 <= scores["Median Score"] <= 1)
        self.assertTrue(0 <= scores["Max Score"] <= 1)
        self.assertTrue(0 <= scores["Min Score"] <= 1)
        self.assertTrue(scores["Standard Deviation"] >= 0)

        # Should have decent scores as most contexts contain relevant information
        self.assertGreater(
            scores["Mean Score"],
            0.5,
            "Contexts with relevant information should have good precision",
        )

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            ContextPrecisionWithoutReference(
                self.vm_dataset,
                user_input_column="invalid_column",
                retrieved_contexts_column="retrieved_contexts",
                response_column="response",
            )

    def test_highly_relevant_contexts(self):
        """Test with contexts that are highly relevant to the response."""
        # Create dataset with highly relevant contexts
        df_relevant = pd.DataFrame(
            {
                "user_input": ["What is the capital of France?"],
                "retrieved_contexts": [
                    [
                        "Paris is the capital city of France.",
                        "Paris is where the French government is located.",
                        "The capital of France has been Paris since 1944.",
                    ]
                ],
                "response": ["The capital of France is Paris."],
            }
        )

        vm_dataset_relevant = vm.init_dataset(
            input_id="relevant_dataset",
            dataset=df_relevant,
            __log=False,
        )

        result = ContextPrecisionWithoutReference(
            vm_dataset_relevant,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
        )

        scores = result[0]["Aggregate Scores"][0]
        # Score should be high for highly relevant contexts
        self.assertGreater(
            scores["Mean Score"],
            0.8,
            "Highly relevant contexts should have very high precision",
        )
