import unittest
import pandas as pd
import numpy as np
import validmind as vm
import plotly.graph_objects as go
from validmind import RawData
from validmind.tests.model_validation.BleuScore import BleuScore


class TestBleuScore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text data
        np.random.seed(42)

        # Create sample texts
        reference_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "Actions speak louder than words.",
        ]

        # Create slightly modified predictions
        predicted_texts = [
            "The brown fox jumps over the lazy dog quickly.",
            "A long journey starts with one step.",
            "To be or not to be, that's the question.",
            "Not all that glitters is gold.",
            "Actions are louder than words.",
        ]

        # Create DataFrame
        self.df = pd.DataFrame(
            {
                "text": reference_texts,
                "target": reference_texts,
                "predictions": predicted_texts,
            }
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

        # Create a mock validmind model to link predictions to
        self.vm_model = vm.init_model(
            input_id="model",
            attributes={
                "architecture": "Mock",
                "language": "Python",
            },
            __log=False,
        )

        # Link predictions to the existing predictions column
        self.vm_dataset.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

    def test_returns_tuple(self):
        """Test if function returns expected tuple structure."""
        result = BleuScore(self.vm_dataset, self.vm_model)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Check if first element is DataFrame
        self.assertIsInstance(result[0], pd.DataFrame)

        # Check if remaining elements include expected number of figures and RawData
        self.assertEqual(len(result), 4)
        for fig in result[1:3]:
            self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(result[3], RawData)

    def test_metrics_dataframe(self):
        """Test if metrics DataFrame has expected structure and values."""
        result_df = BleuScore(self.vm_dataset, self.vm_model)[0]

        # Check expected columns
        expected_columns = [
            "Metric",
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
            "Count",
        ]
        self.assertListEqual(list(result_df.columns), expected_columns)

        # Check if scores are within valid range (0 to 1)
        score_columns = ["Mean Score", "Median Score", "Max Score", "Min Score"]
        for col in score_columns:
            self.assertTrue(all(0 <= score <= 1 for score in result_df[col]))

    def test_figures_properties(self):
        """Test if figures have expected properties."""
        _, *figures, _ = BleuScore(self.vm_dataset, self.vm_model)

        # Check if we have the expected number of figures (2 figures: histogram and bar chart)
        self.assertEqual(len(figures), 2)

        for fig in figures:
            # Check if figure has exactly one trace
            self.assertEqual(len(fig.data), 1)

            # Check if trace type is either histogram or bar
            self.assertIn(fig.data[0].type, ["histogram", "bar"])

    def test_identical_texts(self):
        """Test behavior with identical reference and predicted texts."""
        # Create dataset with identical texts
        identical_texts = ["This is a test."] * 5
        df_identical = pd.DataFrame(
            {
                "text": identical_texts,
                "target": identical_texts,
                "predictions": identical_texts,
            }
        )

        vm_dataset_identical = vm.init_dataset(
            input_id="dataset_identical",
            dataset=df_identical,
            target_column="target",
            __log=False,
        )

        # Link predictions to an existing predictions column
        vm_dataset_identical.assign_predictions(
            model=self.vm_model, prediction_column="predictions"
        )

        result_df, *_ = BleuScore(vm_dataset_identical, self.vm_model)

        # For identical texts, scores should be close to 1
        score = result_df["Mean Score"].iloc[0]
        self.assertGreater(score, 0.9)
