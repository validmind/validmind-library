import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.Duplicates import Duplicates


class TestDuplicates(unittest.TestCase):
    def setUp(self):
        # Create a dataset with 50% duplicates
        df = pd.DataFrame({"A": [1, 2, 1, 2], "B": ["a", "b", "a", "b"]})
        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, feature_columns=["A", "B"], __log=False
        )

        # Create a dataset without duplicates
        df_no_duplicates = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]})
        self.vm_dataset_no_duplicates = vm.init_dataset(
            input_id="test_dataset_no_duplicates",
            dataset=df_no_duplicates,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_duplicates_present(self):
        results, passed = Duplicates(self.vm_dataset)

        # Check structure
        self.assertIn("Duplicate Rows Results for Dataset", results)
        duplicate_results = results["Duplicate Rows Results for Dataset"][0]

        # Check values
        self.assertEqual(duplicate_results["Number of Duplicates"], 2)
        self.assertEqual(duplicate_results["Percentage of Rows (%)"], 50.0)

        # Check test result (should fail with duplicates > threshold)
        self.assertFalse(passed)

    def test_no_duplicates(self):
        results, passed = Duplicates(self.vm_dataset_no_duplicates)

        # Check structure
        self.assertIn("Duplicate Rows Results for Dataset", results)
        duplicate_results = results["Duplicate Rows Results for Dataset"][0]

        # Check values
        self.assertEqual(duplicate_results["Number of Duplicates"], 0)
        self.assertEqual(duplicate_results["Percentage of Rows (%)"], 0.0)

        # Check test result (should pass with no duplicates)
        self.assertTrue(passed)

    def test_custom_threshold(self):
        # Test with a higher threshold that allows some duplicates
        results, passed = Duplicates(self.vm_dataset, min_threshold=3)

        # Should pass because we have 2 duplicates and threshold is 3
        self.assertTrue(passed)
