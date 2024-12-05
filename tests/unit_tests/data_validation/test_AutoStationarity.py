import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.AutoStationarity import AutoStationarity


class TestAutoStationarity(unittest.TestCase):
    def setUp(self):
        # Create a simple time series dataset
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

        np.random.seed(42)  # Set seed for reproducibility

        # Stationary series: mean-reverting AR(1) process
        stationary_data = np.zeros(200)
        for t in range(1, 200):
            stationary_data[t] = 0.5 * stationary_data[t - 1] + np.random.normal(0, 1)

        # Non-stationary series: random walk
        non_stationary = np.cumsum(np.random.normal(0, 1, 200))

        # Non-stationary with trend
        trend_data = np.linspace(0, 10, 200) + np.random.normal(0, 0.1, 200)

        df = pd.DataFrame(
            {
                "stationary": stationary_data,
                "non_stationary": non_stationary,
                "trend": trend_data,
                "with_nans": stationary_data.copy(),  # Copy of stationary with some NaNs
            },
            index=dates,
        )
        # Add some NaN values
        df.loc[df.index[0:2], "with_nans"] = None

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=df,
            __log=False,
        )

    def test_returns_expected_structure(self):
        # Run the function
        result = AutoStationarity(self.vm_dataset)

        # Check if result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("Stationarity Analysis Results", result)
        self.assertIn("Best Integration Order Results", result)

        # Check if tables are DataFrames
        self.assertIsInstance(result["Stationarity Analysis Results"], pd.DataFrame)
        self.assertIsInstance(result["Best Integration Order Results"], pd.DataFrame)

        # Check if tables have expected columns
        analysis_columns = [
            "Variable",
            "Integration Order",
            "Test",
            "p-value",
            "Threshold",
            "Pass/Fail",
            "Decision",
        ]
        best_order_columns = [
            "Variable",
            "Best Integration Order",
            "Test",
            "p-value",
            "Threshold",
            "Decision",
        ]

        self.assertListEqual(
            list(result["Stationarity Analysis Results"].columns), analysis_columns
        )
        self.assertListEqual(
            list(result["Best Integration Order Results"].columns), best_order_columns
        )

    def test_stationarity_detection(self):
        result = AutoStationarity(self.vm_dataset)
        best_orders = result["Best Integration Order Results"]

        # Get integration orders for each series
        stationary_order = best_orders[best_orders["Variable"] == "stationary"][
            "Best Integration Order"
        ].iloc[0]
        non_stationary_order = best_orders[best_orders["Variable"] == "non_stationary"][
            "Best Integration Order"
        ].iloc[0]
        trend_order = best_orders[best_orders["Variable"] == "trend"][
            "Best Integration Order"
        ].iloc[0]

        # Stationary series should need no differencing
        self.assertEqual(stationary_order, 0)

        # Non-stationary (random walk) should need one difference
        self.assertEqual(non_stationary_order, 1)

        # Trend series should need one or two differences
        self.assertGreaterEqual(trend_order, 1)
        self.assertLessEqual(trend_order, 2)

    def test_max_order_parameter(self):
        max_order = 2
        result = AutoStationarity(self.vm_dataset, max_order=max_order)
        analysis_results = result["Stationarity Analysis Results"]

        # Check that no integration order exceeds the maximum
        self.assertTrue(all(analysis_results["Integration Order"] <= max_order))

    def test_threshold_parameter(self):
        custom_threshold = 0.01
        result = AutoStationarity(self.vm_dataset, threshold=custom_threshold)
        analysis_results = result["Stationarity Analysis Results"]

        # Check that threshold is correctly used
        self.assertTrue(all(analysis_results["Threshold"] == custom_threshold))

        # Check Pass/Fail is consistent with threshold
        for _, row in analysis_results.iterrows():
            expected_pass_fail = "Pass" if row["p-value"] < custom_threshold else "Fail"
            self.assertEqual(row["Pass/Fail"], expected_pass_fail)

    def test_handles_nan_values(self):
        # Should run without errors despite NaN values
        result = AutoStationarity(self.vm_dataset)

        # Check if the column with NaNs was processed
        best_orders = result["Best Integration Order Results"]
        self.assertIn("with_nans", best_orders["Variable"].values)

        # The results for the column with NaNs should have valid values
        nan_series_result = best_orders[best_orders["Variable"] == "with_nans"].iloc[0]
        self.assertIsNotNone(nan_series_result["Best Integration Order"])
        self.assertIsNotNone(nan_series_result["p-value"])
        self.assertIsNotNone(nan_series_result["Decision"])
