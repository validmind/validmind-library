import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.RougeScore import RougeScore


class TestRougeScore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample texts with reference and predictions
        reference_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "Actions speak louder than words.",
        ]

        # Create predictions with slight variations
        predicted_texts = [
            "The fast brown fox leaps over the lazy dog.",
            "A long journey starts with one step forward.",
            "To exist or not exist, that is the question.",
            "Everything that shines is not golden.",
            "Deeds speak louder than mere words.",
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
            input_id="rouge_dataset",
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

        # Link predictions
        self.vm_dataset.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

    def test_returns_dataframe(self):
        """Test if function returns expected structure."""
        result_df, *figures, raw_data = RougeScore(self.vm_dataset, self.vm_model)

        # Check return type
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIsInstance(figures, list)
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw_data, RawData)
        # Check expected columns in DataFrame
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

    def test_score_ranges(self):
        """Test if ROUGE scores are within valid range (0 to 1)."""
        result_df, *figures, _ = RougeScore(self.vm_dataset, self.vm_model)

        score_columns = ["Mean Score", "Median Score", "Max Score", "Min Score"]
        for col in score_columns:
            self.assertTrue(all(0 <= score <= 1 for score in result_df[col]))

    def test_metrics_present(self):
        """Test if all expected metrics are present."""
        result_df, *figures, _ = RougeScore(self.vm_dataset, self.vm_model)

        expected_metrics = ["Precision", "Recall", "F1 Score"]
        actual_metrics = result_df["Metric"].tolist()
        self.assertListEqual(sorted(actual_metrics), sorted(expected_metrics))

    def test_figures_properties(self):
        """Test if figures have expected properties."""
        _, *figures, _ = RougeScore(self.vm_dataset, self.vm_model)

        # Should have 6 figures (histogram and bar chart for each metric)
        self.assertEqual(len(figures), 6)

        for fig in figures:
            # Each figure should have one trace
            self.assertEqual(len(fig.data), 1)

            # Each figure should be either histogram or bar
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
            input_id="identical_dataset",
            dataset=df_identical,
            target_column="target",
            __log=False,
        )

        # Link predictions
        vm_dataset_identical.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

        result_df, *figures, _ = RougeScore(vm_dataset_identical, self.vm_model)

        # For identical texts, F1 scores should be 1.0 or very close to 1.0
        f1_score = result_df[result_df["Metric"] == "F1 Score"]["Mean Score"].iloc[0]
        self.assertAlmostEqual(f1_score, 1.0, places=2)

        # Precision and Recall should also be very close to 1.0
        precision = result_df[result_df["Metric"] == "Precision"]["Mean Score"].iloc[0]
        recall = result_df[result_df["Metric"] == "Recall"]["Mean Score"].iloc[0]

        self.assertAlmostEqual(precision, 1.0, places=2)
        self.assertAlmostEqual(recall, 1.0, places=2)

    def test_custom_metric(self):
        """Test if custom ROUGE metric parameter works."""
        result_df, *figures, _ = RougeScore(
            self.vm_dataset, self.vm_model, metric="rouge-2"
        )

        # Should still return DataFrame and figures
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue(all(isinstance(fig, go.Figure) for fig in figures))

        # Check raw data instance
        raw_data = raw_data
        self.assertIsInstance(raw_data, RawData)
