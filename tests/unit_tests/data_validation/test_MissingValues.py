import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.MissingValues import MissingValues


class TestMissingValues(unittest.TestCase):
    def setUp(self):
        # Create test dataset with known missing values
        n_samples = 100

        # Create data with controlled missing values
        data = {
            "no_missing": np.random.rand(n_samples),  # No missing values
            "some_missing": np.random.rand(n_samples),  # Will have some missing values
            "all_missing": np.random.rand(n_samples),  # Will be all missing
        }

        # Insert missing values
        data["some_missing"][:20] = np.nan  # 20% missing
        data["all_missing"][:] = np.nan  # 100% missing

        df = pd.DataFrame(data)

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_missing_values_structure(self):
        # Run the function
        summary, passed, raw_data = MissingValues(self.vm_dataset)

        # Check return types
        self.assertIsInstance(summary, list)
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(raw_data, vm.RawData)

        # Check summary structure
        for column_summary in summary:
            self.assertIn("Column", column_summary)
            self.assertIn("Number of Missing Values", column_summary)
            self.assertIn("Percentage of Missing Values (%)", column_summary)
            self.assertIn("Pass/Fail", column_summary)

    def test_missing_values_counts(self):
        summary, passed, raw_data = MissingValues(self.vm_dataset)

        # Get results for each column
        no_missing = next(s for s in summary if s["Column"] == "no_missing")
        some_missing = next(s for s in summary if s["Column"] == "some_missing")
        all_missing = next(s for s in summary if s["Column"] == "all_missing")

        # Check counts
        self.assertEqual(no_missing["Number of Missing Values"], 0)
        self.assertEqual(some_missing["Number of Missing Values"], 20)
        self.assertEqual(all_missing["Number of Missing Values"], 100)

        # Check percentages
        self.assertEqual(no_missing["Percentage of Missing Values (%)"], 0.0)
        self.assertEqual(some_missing["Percentage of Missing Values (%)"], 20.0)
        self.assertEqual(all_missing["Percentage of Missing Values (%)"], 100.0)

        # Check Pass/Fail status (with default min_threshold=1)
        self.assertEqual(no_missing["Pass/Fail"], "Pass")
        self.assertEqual(some_missing["Pass/Fail"], "Fail")
        self.assertEqual(all_missing["Pass/Fail"], "Fail")

        # Check overall pass/fail
        self.assertFalse(passed)  # Should fail because some columns have missing values

    def test_threshold_parameter(self):
        # Test with higher threshold that allows some missing values
        summary, passed, raw_data = MissingValues(self.vm_dataset, min_threshold=25)

        # Get results
        some_missing = next(s for s in summary if s["Column"] == "some_missing")
        all_missing = next(s for s in summary if s["Column"] == "all_missing")

        # Check Pass/Fail status with new threshold
        self.assertEqual(some_missing["Pass/Fail"], "Pass")  # Should pass (20 < 25)
        self.assertEqual(all_missing["Pass/Fail"], "Fail")  # Should fail (100 > 25)
