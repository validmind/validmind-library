import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind import RawData
from validmind.tests.data_validation.Skewness import Skewness


class TestSkewness(unittest.TestCase):
    def setUp(self):
        # Set consistent size for all columns
        n_samples = 1000

        # Create a dataset with known skewness
        # Normal distribution (low skewness)
        normal_data = np.random.normal(0, 1, n_samples)

        # Right-skewed distribution (high positive skewness)
        skewed_data = np.random.exponential(2, n_samples)

        # Non-numeric column
        categorical = ["A", "B", "C"] * (n_samples // 3)
        if (
            len(categorical) < n_samples
        ):  # Handle case where n_samples isn't divisible by 3
            categorical.extend(["A"] * (n_samples - len(categorical)))

        df = pd.DataFrame(
            {"normal": normal_data, "skewed": skewed_data, "categorical": categorical}
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["normal", "skewed", "categorical"],
            __log=False,
        )

    def test_skewness_threshold(self):
        # Test with default threshold (1)
        results, passed, raw_data = Skewness(self.vm_dataset)

        # Check return types
        self.assertIsInstance(results, dict)
        self.assertIn(passed, [True, False])
        self.assertIsInstance(raw_data, RawData)

        # Check results structure
        results_table = results["Skewness Results for Dataset"]
        self.assertIsInstance(results_table, list)

        # Verify only numeric columns are included
        column_names = {row["Column"] for row in results_table}
        self.assertEqual(column_names, {"normal", "skewed"})

        # Normal distribution should pass, skewed should fail
        for row in results_table:
            if row["Column"] == "normal":
                self.assertEqual(row["Pass/Fail"], "Pass")
            if row["Column"] == "skewed":
                self.assertEqual(row["Pass/Fail"], "Fail")

    def test_custom_threshold(self):
        # Test with very high threshold (all should pass)
        results, passed, raw_data = Skewness(self.vm_dataset, max_threshold=10)
        results_table = results["Skewness Results for Dataset"]

        # All columns should pass with high threshold
        self.assertTrue(passed)
        self.assertTrue(all(row["Pass/Fail"] == "Pass" for row in results_table))
