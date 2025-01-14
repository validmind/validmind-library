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

from validmind.tests.model_validation.ragas.ContextRecall import ContextRecall


class TestContextRecall(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with questions, contexts, and references
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "What is the capital of France?",
                    "Tell me about the theory of relativity.",
                    "Describe the water cycle.",
                ],
                "retrieved_contexts": [
                    [
                        "Paris is the capital of France and its largest city.",  # Full coverage
                        "The city serves as France's political and economic center.",
                    ],
                    [
                        "Einstein developed special relativity in 1905.",  # Partial coverage
                        "The theory explains how space and time are linked.",
                        # Missing E=mc^2 information
                    ],
                    [
                        "Water evaporates from oceans.",  # Missing parts of cycle
                        "Rain falls from clouds.",
                        # Missing condensation and collection steps
                    ],
                ],
                "reference": [
                    "Paris is the capital of France and its largest city. The city serves as France's political and economic center.",  # Should have high recall
                    "Einstein developed special relativity in 1905. The theory explains how space and time are linked. It also led to the famous equation E=mc^2.",  # Should have medium recall
                    "The water cycle involves evaporation from oceans, condensation in clouds, precipitation as rain, and collection in bodies of water.",  # Should have low recall
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="recall_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = ContextRecall(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            reference_column="reference",
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

    def test_recall_scores(self):
        """Test if recall scores reflect coverage of reference information."""
        result = ContextRecall(
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

        # First example has complete coverage, should contribute to high max score
        self.assertGreater(
            scores["Max Score"], 0.8, "Complete coverage should have high recall"
        )

        # Last example has missing information, should contribute to low min score
        self.assertLessEqual(
            scores["Min Score"], 0.5, "Incomplete coverage should have low recall"
        )

    def test_perfect_recall(self):
        """Test with contexts that perfectly match reference."""
        # Create dataset with perfect context coverage
        df_perfect = pd.DataFrame(
            {
                "user_input": ["What is the capital of France?"],
                "retrieved_contexts": [
                    [
                        "Paris is the capital of France.",
                        "It has been the capital since 1944.",
                    ]
                ],
                "reference": [
                    "Paris is the capital of France. It has been the capital since 1944."
                ],
            }
        )

        vm_dataset_perfect = vm.init_dataset(
            input_id="perfect_dataset",
            dataset=df_perfect,
            __log=False,
        )

        result = ContextRecall(
            vm_dataset_perfect,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            reference_column="reference",
        )

        scores = result[0]["Aggregate Scores"][0]
        # Score should be very high for perfect coverage
        self.assertGreater(
            scores["Mean Score"],
            0.9,
            "Perfect context coverage should have very high recall",
        )

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            ContextRecall(
                self.vm_dataset,
                user_input_column="invalid_column",
                retrieved_contexts_column="retrieved_contexts",
                reference_column="reference",
            )
