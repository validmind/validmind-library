import unittest
import pandas as pd
import numpy as np
import validmind as vm
import plotly.graph_objects as go
from validmind.tests.model_validation.BertScore import BertScore


class TestBertScore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with text data
        np.random.seed(42)
        n_samples = 10  # Small sample size for testing

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

        # Link predictions to an existing predictions column
        self.vm_dataset.assign_predictions(
            model=self.vm_model, prediction_column="predictions"
        )

    def test_returns_tuple(self):
        # Run the function
        result = BertScore(self.vm_dataset, self.vm_model)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Should return 7 items: 1 DataFrame and 6 figures (2 for each metric)
        self.assertEqual(len(result), 7)

        # Check if first element is a DataFrame
        self.assertIsInstance(result[0], pd.DataFrame)

        # Check if remaining elements are Plotly Figures
        for fig in result[1:]:
            self.assertIsInstance(fig, go.Figure)

    def test_metrics_dataframe(self):
        result_df, *_ = BertScore(self.vm_dataset, self.vm_model)

        # Check if DataFrame has expected columns
        expected_columns = [
            "Metric",
            "Mean Score",
            "Median Score",
            "Max Score",
            "Min Score",
            "Standard Deviation",
            "Count",
        ]
        self.assertTrue(all(col in result_df.columns for col in expected_columns))

        # Check if all three metrics are present
        expected_metrics = ["Precision", "Recall", "F1 Score"]
        self.assertTrue(
            all(metric in result_df["Metric"].values for metric in expected_metrics)
        )

        # Check if scores are within valid range [0, 1]
        score_columns = ["Mean Score", "Median Score", "Max Score", "Min Score"]
        for col in score_columns:
            self.assertTrue(all(0 <= score <= 1 for score in result_df[col]))

    def test_figures_properties(self):
        _, *figures = BertScore(self.vm_dataset, self.vm_model)

        # Check if we have the expected number of figures (2 per metric)
        self.assertEqual(len(figures), 6)

        for fig in figures:
            # Check if figure has exactly one trace
            self.assertEqual(len(fig.data), 1)

            # Check if trace type is either histogram or bar
            self.assertIn(fig.data[0].type, ["histogram", "bar"])

    def test_identical_texts(self):
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
            input_id="identical_dataset",
            dataset=df_identical,
            target_column="target",
            __log=False,
        )

        # Link predictions to an existing predictions column
        vm_dataset_identical.assign_predictions(
            model=self.vm_model, prediction_column="predictions"
        )

        result_df, *_ = BertScore(vm_dataset_identical, self.vm_model)

        # For identical texts, scores should be close to 1
        for metric in ["Precision", "Recall", "F1 Score"]:
            score = result_df[result_df["Metric"] == metric]["Mean Score"].iloc[0]
            self.assertGreater(score, 0.9)

    def test_length_mismatch_handling(self):
        # Original data has 10 reference texts and 10 predictions
        # We create a mismatch by taking only 9 predictions and adding None
        df = self.df.copy()
        df["predictions_mismatch"] = df["predictions"].iloc[:-1].tolist() + [None]

        # Create vm_dataset with shorter predictions
        vm_dataset_mismatch = vm.init_dataset(
            input_id="mismatch_dataset",
            dataset=df,
            target_column="target",
            __log=False,
        )

        # Link predictions to an existing predictions column
        vm_dataset_mismatch.assign_predictions(
            model=self.vm_model, prediction_column="predictions_mismatch"
        )

        # Should not raise an error
        try:
            result = BertScore(vm_dataset_mismatch, self.vm_model)
            self.assertIsInstance(result, tuple)
        except Exception as e:
            self.fail(f"BertScore raised an unexpected exception: {e}")
