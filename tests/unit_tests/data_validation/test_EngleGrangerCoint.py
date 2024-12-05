import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.EngleGrangerCoint import EngleGrangerCoint
from validmind.errors import SkipTestError


class TestEngleGrangerCoint(unittest.TestCase):
    def setUp(self):
        # Create a time series dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create cointegrated series
        random_walk = np.cumsum(np.random.normal(0, 1, 100))
        cointegrated = random_walk + np.random.normal(0, 0.1, 100)

        # Create non-cointegrated series
        non_cointegrated = np.cumsum(np.random.normal(0, 1, 100))

        df = pd.DataFrame(
            {"A": random_walk, "B": cointegrated, "C": non_cointegrated}, index=dates
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

    def test_engle_granger_structure(self):
        result = EngleGrangerCoint(self.vm_dataset)

        # Check basic structure
        self.assertIn("Cointegration Analysis Results", result)
        self.assertIsInstance(result["Cointegration Analysis Results"], pd.DataFrame)

        # Check columns of the DataFrame
        expected_columns = [
            "Variable 1",
            "Variable 2",
            "Test",
            "p-value",
            "Threshold",
            "Pass/Fail",
            "Decision",
        ]
        self.assertListEqual(
            list(result["Cointegration Analysis Results"].columns), expected_columns
        )

    def test_engle_granger_values(self):
        result = EngleGrangerCoint(self.vm_dataset)
        df_results = result["Cointegration Analysis Results"]

        # Cointegrated pair (A and B) should have lower p-value and pass
        ab_result = df_results[
            (df_results["Variable 1"] == "A") & (df_results["Variable 2"] == "B")
        ].iloc[0]
        self.assertLess(ab_result["p-value"], 0.05)
        self.assertEqual(ab_result["Pass/Fail"], "Pass")
        self.assertEqual(ab_result["Decision"], "Cointegrated")

        # Non-cointegrated pairs (A and C, B and C) should have higher p-value and fail
        ac_result = df_results[
            (df_results["Variable 1"] == "A") & (df_results["Variable 2"] == "C")
        ].iloc[0]
        self.assertGreater(ac_result["p-value"], 0.05)
        self.assertEqual(ac_result["Pass/Fail"], "Fail")
        self.assertEqual(ac_result["Decision"], "Not cointegrated")

        bc_result = df_results[
            (df_results["Variable 1"] == "B") & (df_results["Variable 2"] == "C")
        ].iloc[0]
        self.assertGreater(bc_result["p-value"], 0.05)
        self.assertEqual(bc_result["Pass/Fail"], "Fail")
        self.assertEqual(bc_result["Decision"], "Not cointegrated")

    def test_invalid_index(self):
        # Should raise SkipTestError for non-datetime index
        with self.assertRaises(SkipTestError):
            EngleGrangerCoint(self.vm_dataset_no_date)
