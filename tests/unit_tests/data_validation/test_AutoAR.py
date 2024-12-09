import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.AutoAR import AutoAR


class TestAutoAR(unittest.TestCase):
    def setUp(self):
        # Create a simple time series dataset
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

        # AR(1) process: x(t) = 0.8 * x(t-1) + small_noise
        # Using stronger AR coefficient and smaller noise
        np.random.seed(42)  # Set seed for reproducibility
        ar1_data = np.zeros(200)
        for t in range(1, 200):
            ar1_data[t] = 0.8 * ar1_data[t - 1] + np.random.normal(0, 0.1)

        # AR(2) process: x(t) = 0.6 * x(t-1) + 0.3 * x(t-2) + small_noise
        # Using stronger coefficients and smaller noise
        ar2_data = np.zeros(200)
        for t in range(2, 200):
            ar2_data[t] = (
                0.6 * ar2_data[t - 1] + 0.3 * ar2_data[t - 2] + np.random.normal(0, 0.1)
            )

        df = pd.DataFrame(
            {
                "ar1_process": ar1_data,
                "ar2_process": ar2_data,
                "with_nans": ar1_data.copy(),  # Copy of AR(1) with some NaNs
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
        result = AutoAR(self.vm_dataset, max_ar_order=3)

        # Check if result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("Auto AR Analysis Results", result)
        self.assertIn("Best AR Order Results", result)

        # Check if tables are DataFrames
        self.assertIsInstance(result["Auto AR Analysis Results"], pd.DataFrame)
        self.assertIsInstance(result["Best AR Order Results"], pd.DataFrame)

        # Check if tables have expected columns
        expected_columns = ["Variable", "AR Order", "BIC", "AIC"]
        self.assertListEqual(
            list(result["Auto AR Analysis Results"].columns), expected_columns
        )
        self.assertListEqual(
            list(result["Best AR Order Results"].columns), expected_columns
        )

    def test_ar_order_detection(self):
        result = AutoAR(self.vm_dataset, max_ar_order=3)
        best_orders = result["Best AR Order Results"]

        # Get best AR orders for each process
        ar1_order = best_orders[best_orders["Variable"] == "ar1_process"][
            "AR Order"
        ].iloc[0]
        ar2_order = best_orders[best_orders["Variable"] == "ar2_process"][
            "AR Order"
        ].iloc[0]

        # AR(1) process should have best order of 1
        self.assertEqual(ar1_order, 1)
        # AR(2) process should have best order of 2
        self.assertEqual(ar2_order, 2)

    def test_max_ar_order_parameter(self):
        max_order = 2
        result = AutoAR(self.vm_dataset, max_ar_order=max_order)
        analysis_results = result["Auto AR Analysis Results"]

        # Check that no AR order exceeds the maximum
        self.assertTrue(all(analysis_results["AR Order"] <= max_order))

    def test_handles_nan_values(self):
        # Should run without errors despite NaN values
        result = AutoAR(self.vm_dataset, max_ar_order=3)

        # Check if the column with NaNs was processed
        best_orders = result["Best AR Order Results"]
        self.assertIn("with_nans", best_orders["Variable"].values)

        # The results for the column with NaNs should have valid values
        nan_series_result = best_orders[best_orders["Variable"] == "with_nans"].iloc[0]
        self.assertIsNotNone(nan_series_result["AR Order"])
        self.assertIsNotNone(nan_series_result["BIC"])
        self.assertIsNotNone(nan_series_result["AIC"])
