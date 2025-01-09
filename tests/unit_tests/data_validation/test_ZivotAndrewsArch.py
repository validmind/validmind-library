import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.data_validation.ZivotAndrewsArch import ZivotAndrewsArch
from validmind import RawData


class TestZivotAndrewsArch(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create non-stationary time series (random walk)
        random_walk = np.random.normal(0, 1, 100).cumsum()

        # Create stationary time series
        stationary = np.random.normal(0, 1, 100)

        df = pd.DataFrame(
            {"non_stationary": random_walk, "stationary": stationary}, index=dates
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["non_stationary", "stationary"],
            __log=False,
        )

        # Create dataset without datetime index
        df_no_datetime = pd.DataFrame({"A": range(100), "B": range(100)})

        self.vm_dataset_no_datetime = vm.init_dataset(
            input_id="test_dataset_no_datetime",
            dataset=df_no_datetime,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_zivot_andrews(self):
        result, raw_data = ZivotAndrewsArch(self.vm_dataset)

        # Check return type and structure
        self.assertIsInstance(result, dict)
        self.assertIn("Zivot-Andrews Test Results", result)

        # Check results structure
        za_values = result["Zivot-Andrews Test Results"]
        self.assertIsInstance(za_values, list)
        self.assertEqual(len(za_values), 2)  # One result per column

        # Check required fields in results
        required_fields = ["Variable", "stat", "pvalue", "usedlag", "nobs"]
        for value in za_values:
            for field in required_fields:
                self.assertIn(field, value)

        # Check raw data
        self.assertIsInstance(raw_data, RawData)

    def test_no_datetime_index(self):
        # Should raise SkipTestError when no datetime index present
        with self.assertRaises(SkipTestError):
            ZivotAndrewsArch(self.vm_dataset_no_datetime)
