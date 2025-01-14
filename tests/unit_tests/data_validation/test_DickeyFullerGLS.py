import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.DickeyFullerGLS import DickeyFullerGLS
from validmind.errors import SkipTestError
from validmind import RawData


class TestDickeyFullerGLS(unittest.TestCase):
    def setUp(self):
        # Create a time series dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create different types of time series
        stationary_data = np.random.normal(0, 1, 100)  # Stationary series
        trend_data = np.arange(100) + np.random.normal(
            0, 5, 100
        )  # Non-stationary series with trend

        df = pd.DataFrame(
            {
                "stationary": stationary_data,
                "trend": trend_data,
            },
            index=dates,
        )

        # Initialize VMDataset
        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

        # Create a dataset without datetime index for testing error case
        df_no_date = pd.DataFrame({"col1": range(100), "col2": range(100)})
        self.vm_dataset_no_date = vm.init_dataset(
            input_id="test_dataset_no_date", dataset=df_no_date, __log=False
        )

    def test_dfgls_structure(self):
        result, raw_data = DickeyFullerGLS(self.vm_dataset)

        # Check basic structure
        self.assertIn("DFGLS Test Results", result)
        self.assertIsInstance(result["DFGLS Test Results"], list)

        # Check raw data
        self.assertIsInstance(raw_data, RawData)

        # Check results for each variable
        for var_result in result["DFGLS Test Results"]:
            self.assertIn("Variable", var_result)
            self.assertIn("stat", var_result)
            self.assertIn("pvalue", var_result)
            self.assertIn("usedlag", var_result)
            self.assertIn("nobs", var_result)

    def test_dfgls_values(self):
        result, _ = DickeyFullerGLS(self.vm_dataset)
        results_dict = {item["Variable"]: item for item in result["DFGLS Test Results"]}

        # Stationary series should have lower p-value
        self.assertIsNotNone(results_dict["stationary"]["pvalue"])

        # Non-stationary (trend) series should have higher p-value
        self.assertIsNotNone(results_dict["trend"]["pvalue"])

        # The trend series should be less stationary than the random series
        self.assertGreater(
            results_dict["trend"]["pvalue"], results_dict["stationary"]["pvalue"]
        )

    def test_invalid_index(self):
        # Should raise SkipTestError for non-datetime index
        with self.assertRaises(SkipTestError):
            DickeyFullerGLS(self.vm_dataset_no_date)
