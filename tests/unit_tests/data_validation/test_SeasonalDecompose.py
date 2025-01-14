import unittest
import pandas as pd
import numpy as np
import validmind as vm
import plotly.graph_objects as go
from validmind import RawData
from validmind.tests.data_validation.SeasonalDecompose import SeasonalDecompose
from validmind.errors import SkipTestError


class TestSeasonalDecompose(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset with seasonal pattern
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        seasonal_pattern = np.sin(np.linspace(0, 4 * np.pi, 100))  # 2 complete cycles
        trend = np.linspace(0, 2, 100)  # upward trend
        noise = np.random.normal(0, 0.1, 100)

        df = pd.DataFrame(
            {
                "feature1": seasonal_pattern + trend + noise,
                "feature2": seasonal_pattern * 2 + trend + noise,
            },
            index=dates,
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["feature1", "feature2"],
            __log=False,
        )

        # Create dataset with non-finite values
        df_with_nan = df.copy()
        df_with_nan.iloc[0:10, 0] = np.nan
        self.vm_dataset_with_nan = vm.init_dataset(
            input_id="test_dataset_with_nan",
            dataset=df_with_nan,
            feature_columns=["feature1", "feature2"],
            __log=False,
        )

    def test_seasonal_decompose(self):
        result = SeasonalDecompose(self.vm_dataset)

        # Check that we get the correct number of figures (one per feature + RawData)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        # Check that outputs are plotly figures with correct subplots
        figures = result[:-1]  # all but the last item, which is RawData
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)
            # Should have 6 subplots: Observed, Trend, Seasonal, Residuals,
            # Histogram, and Q-Q plot
            self.assertEqual(len(fig.data), 7)  # 6 plots + 1 QQ line

        # Check the last element in the returned result is a RawData instance
        self.assertIsInstance(result[-1], RawData)

    def test_seasonal_decompose_with_nan(self):
        # Should still work with NaN values
        result = SeasonalDecompose(self.vm_dataset_with_nan)

        # Expect 2 plots and 1 RawData object
        self.assertEqual(len(result), 3)

    def test_seasonal_decompose_models(self):
        # Test additive model (should work with any data)
        result_add = SeasonalDecompose(self.vm_dataset, seasonal_model="additive")

        # Expect 2 plots and 1 RawData object
        self.assertEqual(len(result_add), 3)

        # Test multiplicative model (should raise ValueError for data with zero/negative values)
        with self.assertRaises(ValueError) as context:
            SeasonalDecompose(self.vm_dataset, seasonal_model="multiplicative")

        # Verify the error message
        self.assertIn(
            "Multiplicative seasonality is not appropriate for zero and negative values",
            str(context.exception),
        )
