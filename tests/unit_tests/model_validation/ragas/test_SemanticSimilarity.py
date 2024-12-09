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

from validmind.tests.model_validation.ragas.SemanticSimilarity import SemanticSimilarity


class TestSemanticSimilarity(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data with varying levels of semantic similarity
        self.df = pd.DataFrame(
            {
                "response": [
                    "The capital of France is Paris.",  # Identical meaning
                    "Paris serves as France's capital city.",  # Same meaning, different words
                    "Paris is a city in France with many museums.",  # Related but different focus
                ],
                "reference": [
                    "The capital of France is Paris.",
                    "The capital city of France is Paris.",
                    "Paris is the capital of France and the largest city in the country.",
                ],
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="similarity_dataset",
            dataset=self.df,
            __log=False,
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = SemanticSimilarity(
            self.vm_dataset, response_column="response", reference_column="reference"
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

    def test_similarity_scores(self):
        """Test if similarity scores reflect semantic closeness."""
        result = SemanticSimilarity(
            self.vm_dataset, response_column="response", reference_column="reference"
        )

        scores = result[0]["Aggregate Scores"][0]

        # Check that scores exist and are numeric
        self.assertIsInstance(scores["Mean Score"], (int, float))
        self.assertIsInstance(scores["Median Score"], (int, float))
        self.assertIsInstance(scores["Max Score"], (int, float))
        self.assertIsInstance(scores["Min Score"], (int, float))
        self.assertIsInstance(scores["Standard Deviation"], (int, float))
        self.assertEqual(scores["Count"], len(self.df))

        # First example is identical, should have high similarity
        self.assertGreater(
            scores["Max Score"],
            scores["Mean Score"],
            "Identical meanings should have higher than average similarity",
        )

        # Last example is less similar, should have lower score
        self.assertLess(
            scores["Min Score"],
            scores["Mean Score"],
            "Less similar meanings should have lower than average similarity",
        )

    def test_perfect_similarity(self):
        """Test with identical response and reference."""
        # Create dataset with perfect matches
        df_perfect = pd.DataFrame(
            {
                "response": ["The speed of light is 299,792,458 meters per second."],
                "reference": ["The speed of light is 299,792,458 meters per second."],
            }
        )

        vm_dataset_perfect = vm.init_dataset(
            input_id="perfect_dataset",
            dataset=df_perfect,
            __log=False,
        )

        result = SemanticSimilarity(
            vm_dataset_perfect, response_column="response", reference_column="reference"
        )

        scores = result[0]["Aggregate Scores"][0]
        # Score should be very high for identical text
        self.assertGreater(
            scores["Mean Score"],
            0.95,
            "Identical text should have near-perfect similarity",
        )

    def test_invalid_columns(self):
        """Test if function raises error with invalid column names."""
        with self.assertRaises(KeyError):
            SemanticSimilarity(
                self.vm_dataset,
                response_column="invalid_column",
                reference_column="reference",
            )
