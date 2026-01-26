import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.RegardScore import RegardScore


class TestRegardScore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample texts with varying sentiment
        reference_texts = [
            "I love working with this amazing team!",
            "The service was absolutely terrible.",
            "The weather is quite nice today.",
            "I can't believe how wonderful this product is!",
            "This is just a normal, average day.",
        ]

        # Create predictions with slight variations
        predicted_texts = [
            "I really enjoy working with this great team!",
            "The service was really bad.",
            "Today's weather is pleasant.",
            "This product is absolutely amazing!",
            "It's just another ordinary day.",
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
            input_id="regard_dataset",
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

    def test_returns_tuple(self):
        """Test if function returns expected tuple structure."""
        result = RegardScore(self.vm_dataset, self.vm_model)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Check if first element is DataFrame
        self.assertIsInstance(result[0], pd.DataFrame)

        # Check if remaining elements are figures
        for fig in result[1:-1]:
            self.assertIsInstance(fig, go.Figure)

        # Check if last element is RawData
        self.assertIsInstance(result[-1], RawData)

    def test_metrics_dataframe(self):
        """Test if metrics DataFrame has expected structure and values."""
        result_df = RegardScore(self.vm_dataset, self.vm_model)[0]

        # Check expected columns
        expected_columns = [
            "Metric",
            "Category",
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
        result_df, *figures, _ = RegardScore(self.vm_dataset, self.vm_model)

        # Calculate expected number of figures based on actual categories
        # Each category gets 2 figures (histogram + bar chart) for both true and predicted texts
        # Get unique categories from the result dataframe
        categories = result_df["Category"].unique()
        num_categories = len(categories)
        # Expected: 2 figures per category (histogram + bar) for true text + 2 figures per category for predicted text
        expected_num_figures = num_categories * 2 * 2

        # Check if we have the expected number of figures
        self.assertEqual(
            len(figures),
            expected_num_figures,
            msg=f"Expected {expected_num_figures} figures (2 per category for true and predicted, {num_categories} categories), but got {len(figures)}",
        )

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

        result_df, *_ = RegardScore(vm_dataset_identical, self.vm_model)

        # Get neutral scores for true and predicted texts
        true_neutral = result_df[
            (result_df["Metric"] == "True Text Regard")
            & (result_df["Category"] == "neutral")
        ]["Mean Score"].iloc[0]

        pred_neutral = result_df[
            (result_df["Metric"] == "Predicted Text Regard")
            & (result_df["Category"] == "neutral")
        ]["Mean Score"].iloc[0]

        # Scores should be very close for identical texts
        self.assertAlmostEqual(
            true_neutral,
            pred_neutral,
            places=2,
            msg=f"Neutral scores differ: true={true_neutral}, pred={pred_neutral}",
        )
