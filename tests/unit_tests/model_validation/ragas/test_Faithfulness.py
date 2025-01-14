import os
import dotenv
import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind import RawData

# Load environment variables at the start of the test file
dotenv.load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is not set")

from validmind.tests.model_validation.ragas.Faithfulness import Faithfulness


class TestFaithfulness(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with contexts and responses of varying faithfulness
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "How tall is the Eiffel Tower and when was it built?",
                    "Tell me about the creation of Python programming language.",
                    "What can you tell me about Mount Everest?",
                ],
                "retrieved_contexts": [
                    [
                        "The Eiffel Tower is 324 meters tall.",
                        "It was completed in 1889 for the World's Fair.",
                    ],
                    [
                        "Python was created by Guido van Rossum.",
                        "The language was first released in 1991.",
                    ],
                    [
                        "Mount Everest is 8,848 meters high.",
                        "It's located in the Himalayas between Nepal and Tibet.",
                    ],
                ],
                "response": [
                    "The Eiffel Tower, completed in 1889, stands 324 meters tall.",  # Faithful - only uses context info
                    "Python was created by Guido van Rossum in 1991. It's now the most popular programming language.",  # Partially faithful - adds claim not in context
                    "Mount Everest is the tallest mountain in Asia and has claimed many lives.",  # Unfaithful - makes claims not supported by context
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="faithfulness_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = Faithfulness(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
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
        self.assertIsInstance(result[3], RawData)

    def test_faithfulness_scores(self):
        """Test if faithfulness scores reflect response accuracy to contexts."""
        result = Faithfulness(
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
        self.assertEqual(scores["Count"], len(self.df))

        # First example is completely faithful, should contribute to high max score
        self.assertGreater(
            scores["Max Score"],
            0.8,
            "Completely faithful response should have high score",
        )

        # Last example has unsupported claims, should contribute to low min score
        self.assertLess(
            scores["Min Score"], 0.5, "Unfaithful response should have low score"
        )

    def test_perfect_faithfulness(self):
        """Test with responses that are perfectly faithful to contexts."""
        # Create dataset with perfectly faithful responses
        df_faithful = pd.DataFrame(
            {
                "user_input": ["Tell me about Earth's orbit around the Sun."],
                "retrieved_contexts": [
                    [
                        "The Earth orbits the Sun at an average distance of 93 million miles.",
                        "This orbit takes 365.25 days to complete.",
                    ]
                ],
                "response": [
                    "The Earth takes 365.25 days to orbit the Sun, which it does at an average distance of 93 million miles."
                ],
            }
        )

        vm_dataset_faithful = vm.init_dataset(
            input_id="faithful_dataset",
            dataset=df_faithful,
            __log=False,
        )

        result = Faithfulness(
            vm_dataset_faithful,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
        )

        scores = result[0]["Aggregate Scores"][0]
        # Score should be very high for perfectly faithful response
        self.assertGreater(
            scores["Mean Score"],
            0.9,
            "Perfectly faithful response should have very high score",
        )

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            Faithfulness(
                self.vm_dataset,
                retrieved_contexts_column="invalid_column",
                response_column="response",
            )
