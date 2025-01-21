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

from validmind.tests.model_validation.ragas.ResponseRelevancy import ResponseRelevancy


class TestResponseRelevancy(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with varying levels of response relevancy
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "What is the capital of France?",
                    "What is the speed of light?",
                    "Who wrote Romeo and Juliet?",
                ],
                "response": [
                    "Paris is the capital of France.",  # Highly relevant
                    "Light travels at different speeds through different mediums, but in a vacuum it moves at approximately 299,792,458 meters per second.",  # Relevant but verbose
                    "Shakespeare wrote many plays including Hamlet and Macbeth.",  # Partially relevant, misses direct answer
                ],
                "retrieved_contexts": [
                    ["Paris is the capital city of France.", "France is in Europe."],
                    [
                        "The speed of light in vacuum is 299,792,458 m/s.",
                        "Light is electromagnetic radiation.",
                    ],
                    [
                        "William Shakespeare wrote many famous plays.",
                        "Romeo and Juliet was written in the 16th century.",
                    ],
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="relevancy_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = ResponseRelevancy(
            self.vm_dataset,
            user_input_column="user_input",
            response_column="response",
            retrieved_contexts_column="retrieved_contexts",
        )

        # Check return types
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)  # dict, 2 figures, and raw data

        # Check dictionary structure
        self.assertIsInstance(result[0], dict)
        self.assertIn("Aggregate Scores", result[0])
        self.assertIsInstance(result[0]["Aggregate Scores"], list)
        self.assertEqual(len(result[0]["Aggregate Scores"]), 1)

        # Check figures
        self.assertIsInstance(result[1], go.Figure)  # Histogram
        self.assertIsInstance(result[2], go.Figure)  # Box plot

        # Check raw data
        self.assertIsInstance(result[3], vm.RawData)

    def test_relevancy_scores(self):
        """Test if relevancy scores reflect response quality."""
        result = ResponseRelevancy(
            self.vm_dataset, user_input_column="user_input", response_column="response"
        )

        scores = result[0]["Aggregate Scores"][0]

        # Check score ranges
        self.assertTrue(0 <= scores["Mean Score"] <= 1)
        self.assertTrue(0 <= scores["Median Score"] <= 1)
        self.assertTrue(0 <= scores["Max Score"] <= 1)
        self.assertTrue(0 <= scores["Min Score"] <= 1)
        self.assertTrue(scores["Standard Deviation"] >= 0)
        self.assertEqual(scores["Count"], len(self.df))

        # First example is highly relevant, should contribute to high max score
        self.assertGreater(
            scores["Max Score"], 0.8, "Highly relevant response should have high score"
        )

        # Mean should be decent as responses are generally relevant
        self.assertGreater(
            scores["Mean Score"], 0.5, "Average relevancy should be reasonable"
        )

    def test_without_contexts(self):
        """Test that function works without retrieved_contexts."""
        # Create dataset without contexts
        df_no_contexts = pd.DataFrame(
            {
                "user_input": ["What is the capital of France?"],
                "response": ["Paris is the capital of France."],
            }
        )

        vm_dataset_no_contexts = vm.init_dataset(
            input_id="no_contexts_dataset",
            dataset=df_no_contexts,
            __log=False,
        )

        result = ResponseRelevancy(
            vm_dataset_no_contexts,
            user_input_column="user_input",
            response_column="response",
        )

        # Should work without contexts
        self.assertIsInstance(result, tuple)
        scores = result[0]["Aggregate Scores"][0]
        self.assertTrue(0 <= scores["Mean Score"] <= 1)

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            ResponseRelevancy(
                self.vm_dataset,
                user_input_column="invalid_column",
                response_column="response",
            )
