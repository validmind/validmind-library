import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.AutoMA import AutoMA


class TestAutoMA(unittest.TestCase):
    def setUp(self):
        # Create a simple time series dataset
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

        # MA(1) process: x(t) = e(t) + 0.8 * e(t-1)
        np.random.seed(42)  # Set seed for reproducibility
        noise = np.random.normal(0, 0.1, 200)
        ma1_data = np.zeros(200)
        for t in range(1, 200):
            ma1_data[t] = noise[t] + 0.8 * noise[t - 1]

        # MA(2) process: x(t) = e(t) + 0.6 * e(t-1) + 0.3 * e(t-2)
        ma2_data = np.zeros(200)
        for t in range(2, 200):
            ma2_data[t] = noise[t] + 0.6 * noise[t - 1] + 0.3 * noise[t - 2]

        df = pd.DataFrame(
            {
                "ma1_process": ma1_data,
                "ma2_process": ma2_data,
                "with_nans": ma1_data.copy(),  # Copy of MA(1) with some NaNs
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
        result = AutoMA(self.vm_dataset, max_ma_order=3)

        # Check if result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("Auto MA Analysis Results", result)
        self.assertIn("Best MA Order Results", result)

        # Check if tables are DataFrames
        self.assertIsInstance(result["Auto MA Analysis Results"], pd.DataFrame)
        self.assertIsInstance(result["Best MA Order Results"], pd.DataFrame)

        # Check if tables have expected columns
        expected_columns = ["Variable", "MA Order", "BIC", "AIC"]
        self.assertListEqual(
            list(result["Auto MA Analysis Results"].columns), expected_columns
        )
        self.assertListEqual(
            list(result["Best MA Order Results"].columns), expected_columns
        )

    def test_ma_order_detection(self):
        result = AutoMA(self.vm_dataset, max_ma_order=3)
        best_orders = result["Best MA Order Results"]

        # Get best MA orders for each process
        ma1_order = best_orders[best_orders["Variable"] == "ma1_process"][
            "MA Order"
        ].iloc[0]
        ma2_order = best_orders[best_orders["Variable"] == "ma2_process"][
            "MA Order"
        ].iloc[0]

        # MA(1) process should have best order of 1
        self.assertEqual(ma1_order, 1)
        # MA(2) process should have best order of 2
        self.assertEqual(ma2_order, 2)

    def test_max_ma_order_parameter(self):
        max_order = 2
        result = AutoMA(self.vm_dataset, max_ma_order=max_order)
        analysis_results = result["Auto MA Analysis Results"]

        # Check that no MA order exceeds the maximum
        self.assertTrue(all(analysis_results["MA Order"] <= max_order))

    def test_handles_nan_values(self):
        # Should run without errors despite NaN values
        result = AutoMA(self.vm_dataset, max_ma_order=3)
