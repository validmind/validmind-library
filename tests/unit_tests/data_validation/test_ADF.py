import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.ADF import ADF


class TestADF(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducible results
        np.random.seed(42)

        # Create a simple time series dataset with larger size for more stable results
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

        # Create more clearly stationary vs non-stationary series
        self.df = pd.DataFrame(
            {
                "stationary": np.random.normal(0, 1, 200),  # Stationary series
                "non_stationary": np.cumsum(
                    np.random.normal(0, 1, 200)
                ),  # Random walk (non-stationary)
                "with_nans": np.random.normal(0, 1, 200),  # Series with NaN values
            },
            index=dates,
        )
        # Add some NaN values
        self.df.loc[self.df.index[0:2], "with_nans"] = None

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

    def test_returns_expected_structure(self):
        # Run the function
        result = ADF(self.vm_dataset)

        # Check if result is a dictionary with the expected key
        self.assertIsInstance(result, dict)
        self.assertIn("ADF Test Results for Each Feature", result)

        # Get the results table
        table = result["ADF Test Results for Each Feature"]

        # Check if table is a DataFrame
        self.assertIsInstance(table, pd.DataFrame)

        # Check if table has expected columns
        expected_columns = [
            "Feature",
            "ADF Statistic",
            "P-Value",
            "Used Lag",
            "Number of Observations",
            "Critical Values",
            "IC Best",
        ]
        self.assertListEqual(list(table.columns), expected_columns)

        # Check if table has one row per feature
        self.assertEqual(len(table), len(self.df.columns))

    def test_stationary_vs_nonstationary(self):
        result = ADF(self.vm_dataset)
        table = result["ADF Test Results for Each Feature"]

        # Get results for both series
        stationary_result = table[table["Feature"] == "stationary"].iloc[0]
        nonstationary_result = table[table["Feature"] == "non_stationary"].iloc[0]

        # Instead of comparing p-values directly (which can be flaky),
        # check that the ADF statistic for stationary series is more negative
        # (indicating stronger evidence of stationarity)
        self.assertLess(
            stationary_result["ADF Statistic"],
            nonstationary_result["ADF Statistic"],
            "Stationary series should have a more negative ADF statistic than non-stationary series",
        )

        # Also check that both p-values are reasonable (between 0 and 1)
        self.assertGreaterEqual(stationary_result["P-Value"], 0)
        self.assertLessEqual(stationary_result["P-Value"], 1)
        self.assertGreaterEqual(nonstationary_result["P-Value"], 0)
        self.assertLessEqual(nonstationary_result["P-Value"], 1)

    def test_raises_error_for_non_datetime_index(self):
        # Create dataset with non-datetime index
        df_wrong_index = self.df.reset_index()
        vm_dataset_wrong_index = vm.init_dataset(
            input_id="dataset",
            dataset=df_wrong_index,
            __log=False,
        )

        # Check if it raises ValueError
        with self.assertRaises(ValueError) as context:
            ADF(vm_dataset_wrong_index)

        self.assertEqual(
            str(context.exception),
            "Dataset index must be a datetime or period index for time series analysis.",
        )

    def test_handles_nan_values(self):
        # Should run without errors despite NaN values
        result = ADF(self.vm_dataset)
        table = result["ADF Test Results for Each Feature"]

        # Check if the column with NaNs was processed
        self.assertIn("with_nans", table["Feature"].values)

        # The row for the column with NaNs should have valid results
        nan_series_result = table[table["Feature"] == "with_nans"].iloc[0]
        self.assertIsNotNone(nan_series_result["ADF Statistic"])
        self.assertIsNotNone(nan_series_result["P-Value"])
