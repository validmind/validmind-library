import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.model_validation.TokenDisparity import TokenDisparity


class TestTokenDisparity(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample texts with varying token counts
        reference_texts = [
            "This is a short sentence.",
            "This sentence has exactly seven words here.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing token counts in this example text.",
            "One two three four five words.",
        ]

        # Create predictions with slightly different token counts
        predicted_texts = [
            "That is a brief sentence.",
            "This sentence contains exactly eight words here.",
            "A quick brown fox jumped over lazy dogs.",
            "Testing the token counts example.",
            "One two three four five.",
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
            input_id="token_dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

        # Initialize ValidMind model
        self.vm_model = vm.init_model(
            input_id="token_model",
            attributes={
                "architecture": "TextModel",
                "language": "Python",
            },
            __log=False,
        )

        # Link predictions
        self.vm_dataset.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        result = TokenDisparity(self.vm_dataset, self.vm_model)

        # Check return types
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], pd.DataFrame)

        # Check raw data is an instance of RawData
        self.assertIsInstance(result[-1], vm.RawData)

        # Check all figures are Plotly figures
        for fig in result[1:-1]:
            self.assertIsInstance(fig, go.Figure)

    def test_results_dataframe(self):
        """Test if results DataFrame has expected structure."""
        result_df = TokenDisparity(self.vm_dataset, self.vm_model)[0]

        # Check columns
        expected_columns = [
            "Metric",
            "Mean Count",
            "Median Count",
            "Max Count",
            "Min Count",
            "Standard Deviation",
            "Count",
        ]
        self.assertListEqual(list(result_df.columns), expected_columns)

        # Check metrics
        expected_metrics = ["Reference Tokens", "Generated Tokens"]
        self.assertListEqual(result_df["Metric"].tolist(), expected_metrics)

        # Check counts are non-negative
        numeric_columns = ["Mean Count", "Median Count", "Max Count", "Min Count"]
        for col in numeric_columns:
            self.assertTrue(all(count >= 0 for count in result_df[col]))

    def test_identical_texts(self):
        """Test behavior with identical reference and predicted texts."""
        # Create dataset with identical texts
        identical_texts = ["This is a test sentence with six words."] * 5

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

        result_df = TokenDisparity(vm_dataset_identical, self.vm_model)[0]

        # Token counts should be identical for reference and generated texts
        ref_mean = result_df[result_df["Metric"] == "Reference Tokens"][
            "Mean Count"
        ].iloc[0]
        gen_mean = result_df[result_df["Metric"] == "Generated Tokens"][
            "Mean Count"
        ].iloc[0]
        self.assertEqual(ref_mean, gen_mean)

        # Standard deviation should be 0 for identical texts
        ref_std = result_df[result_df["Metric"] == "Reference Tokens"][
            "Standard Deviation"
        ].iloc[0]
        gen_std = result_df[result_df["Metric"] == "Generated Tokens"][
            "Standard Deviation"
        ].iloc[0]
        self.assertEqual(ref_std, 0)
        self.assertEqual(gen_std, 0)
