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

from validmind.tests.model_validation.ragas.NoiseSensitivity import NoiseSensitivity


class TestNoiseSensitivity(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with varying levels of noise in contexts
        self.df = pd.DataFrame(
            {
                "user_input": [
                    "What can you tell me about Earth and its atmosphere?",
                    "What is Python and who created it?",
                    "What is the speed of light?",
                ],
                "retrieved_contexts": [
                    [
                        "The Earth is the third planet from the Sun.",  # Relevant
                        "The Earth's atmosphere is mostly nitrogen.",  # Relevant
                    ],
                    [
                        "Python is a programming language.",  # Relevant
                        "Bananas are a good source of potassium.",  # Irrelevant noise
                        "Python was created by Guido van Rossum.",  # Relevant
                    ],
                    [
                        "Shakespeare wrote many sonnets.",  # Irrelevant noise
                        "The speed of light is 299,792,458 m/s.",  # Relevant
                        "Light travels faster than sound.",  # Relevant
                    ],
                ],
                "response": [
                    "The Earth is the third planet from the Sun and has an atmosphere composed mostly of nitrogen.",  # Uses only relevant info
                    "Python is a programming language created by Guido van Rossum.",  # Ignores noise
                    "Light travels at 299,792,458 meters per second, which is faster than sound. Shakespeare was a great writer.",  # Includes noise
                ],
                "reference": [
                    "The Earth is the third planet from the Sun with a nitrogen-rich atmosphere.",
                    "Python is a programming language developed by Guido van Rossum.",
                    "The speed of light is 299,792,458 meters per second, exceeding the speed of sound.",
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="noise_sensitivity_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = NoiseSensitivity(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
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
        self.assertIsInstance(result[3], RawData)  # Raw data check

    def test_noise_sensitivity_scores(self):
        """Test if noise sensitivity scores reflect response quality."""
        result = NoiseSensitivity(
            self.vm_dataset,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
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

        # Lower scores are better (less sensitive to noise)
        self.assertLess(
            scores["Mean Score"],
            0.5,
            "Responses should generally show low sensitivity to noise",
        )

        # Even the worst case should show reasonable noise handling
        self.assertLess(
            scores["Max Score"], 0.7, "Even highest sensitivity should be moderate"
        )

    def test_focus_parameter(self):
        """Test both focus parameter values."""
        # Create dataset with perfect context coverage
        df_focus = pd.DataFrame(
            {
                "user_input": ["What is the capital of France?"],
                "retrieved_contexts": [
                    [
                        "Paris is the capital of France.",  # Relevant
                        "The Eiffel Tower is in Paris.",  # Relevant
                        "Rome is in Italy.",  # Irrelevant noise
                    ]
                ],
                "response": ["Paris is the capital of France."],
                "reference": ["The capital of France is Paris."],
            }
        )

        vm_dataset_focus = vm.init_dataset(
            input_id="focus_dataset",
            dataset=df_focus,
            __log=False,
        )

        # Test with focus='relevant'
        result_relevant = NoiseSensitivity(
            vm_dataset_focus,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
            reference_column="reference",
            focus="relevant",
        )
        self.assertIsInstance(result_relevant, tuple)

        # Test with focus='irrelevant'
        result_irrelevant = NoiseSensitivity(
            vm_dataset_focus,
            user_input_column="user_input",
            retrieved_contexts_column="retrieved_contexts",
            response_column="response",
            reference_column="reference",
            focus="irrelevant",
        )
        self.assertIsInstance(result_irrelevant, tuple)

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            NoiseSensitivity(
                self.vm_dataset,
                user_input_column="user_input",
                retrieved_contexts_column="invalid_column",
                response_column="response",
                reference_column="reference",
            )
