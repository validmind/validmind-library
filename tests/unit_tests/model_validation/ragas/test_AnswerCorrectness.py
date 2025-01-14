import os
import dotenv

dotenv.load_dotenv()
if not "OPENAI_API_KEY" in os.environ:
    raise ValueError("OPENAI_API_KEY is not set")

import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind import RawData

from validmind.tests.model_validation.ragas.AnswerCorrectness import AnswerCorrectness


class TestAnswerCorrectness(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.df = pd.DataFrame(
            {
                "question": [
                    "What is the capital of France?",
                    "What is the largest planet in our solar system?",
                ],
                "answer": [
                    "Paris is the capital of France.",
                    "Jupiter is the largest planet in our solar system.",
                ],
                "ground_truth": [
                    "The capital of France is Paris.",
                    "Jupiter is the largest planet in the solar system.",
                ],
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="qa_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = AnswerCorrectness(
            self.vm_dataset,
            user_input_column="question",
            response_column="answer",
            reference_column="ground_truth",
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)  # dict, 2 figures, and raw data

        self.assertIsInstance(result[0], dict)
        self.assertIn("Aggregate Scores", result[0])
        self.assertIsInstance(result[0]["Aggregate Scores"], list)
        self.assertEqual(len(result[0]["Aggregate Scores"]), 1)

        self.assertIsInstance(result[1], go.Figure)  # Histogram
        self.assertIsInstance(result[2], go.Figure)  # Box plot

        self.assertIsInstance(result[3], RawData)  # Raw data

    def test_aggregate_scores(self):
        """Test if aggregate scores have expected structure and values."""
        result = AnswerCorrectness(
            self.vm_dataset,
            user_input_column="question",
            response_column="answer",
            reference_column="ground_truth",
        )

        scores = result[0]["Aggregate Scores"][0]

        expected_keys = [
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
            "Count",
        ]
        self.assertListEqual(list(scores.keys()), expected_keys)

        self.assertTrue(0 <= scores["Mean Score"] <= 1)
        self.assertTrue(0 <= scores["Median Score"] <= 1)
        self.assertTrue(0 <= scores["Max Score"] <= 1)
        self.assertTrue(0 <= scores["Min Score"] <= 1)
        self.assertTrue(scores["Standard Deviation"] >= 0)
        self.assertEqual(scores["Count"], len(self.df))

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            AnswerCorrectness(
                self.vm_dataset,
                user_input_column="invalid_column",
                response_column="answer",
                reference_column="ground_truth",
            )
