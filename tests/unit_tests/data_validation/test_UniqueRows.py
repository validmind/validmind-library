import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.UniqueRows import UniqueRows


class TestUniqueRows(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with known uniqueness
        n_samples = 100

        # Dataset with high uniqueness (all rows unique)
        df_unique = pd.DataFrame(
            {"A": range(n_samples), "B": [i * 2 for i in range(n_samples)]}
        )

        self.vm_dataset_unique = vm.init_dataset(
            input_id="test_dataset_unique",
            dataset=df_unique,
            feature_columns=["A", "B"],
            __log=False,
        )

        # Dataset with low uniqueness (many duplicates)
        df_duplicates = pd.DataFrame(
            {"A": [1, 2] * (n_samples // 2), "B": [3, 4] * (n_samples // 2)}
        )

        self.vm_dataset_duplicates = vm.init_dataset(
            input_id="test_dataset_duplicates",
            dataset=df_duplicates,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_unique_rows_default_threshold(self):
        # Test dataset with high uniqueness
        results, passed = UniqueRows(self.vm_dataset_unique)

        # Check return types
        self.assertIsInstance(results, list)
        self.assertIsInstance(passed, bool)

        # Check results structure
        for result in results:
            self.assertIn("Column", result)
            self.assertIn("Number of Unique Values", result)
            self.assertIn("Percentage of Unique Values (%)", result)
            self.assertIn("Pass/Fail", result)

        # Should pass with 100% unique values
        self.assertTrue(passed)
        self.assertTrue(all(row["Pass/Fail"] == "Pass" for row in results))

    def test_low_uniqueness(self):
        # Test dataset with low uniqueness
        results, passed = UniqueRows(self.vm_dataset_duplicates)

        # Check return types
        self.assertIsInstance(results, list)
        self.assertIsInstance(passed, bool)

        # Check results structure
        for result in results:
            self.assertIn("Column", result)
            self.assertIn("Number of Unique Values", result)
            self.assertIn("Percentage of Unique Values (%)", result)
            self.assertIn("Pass/Fail", result)

        # Should fail with low uniqueness
        self.assertFalse(passed)
        self.assertTrue(any(row["Pass/Fail"] == "Fail" for row in results))
