import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.TooManyZeroValues import TooManyZeroValues


class TestTooManyZeroValues(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with known zero distributions
        n_samples = 1000

        # Column with few zeros (1%)
        few_zeros = np.random.normal(1, 0.5, n_samples)
        few_zeros[:10] = 0  # 10 zeros = 1%

        # Column with many zeros (5%)
        many_zeros = np.random.normal(1, 0.5, n_samples)
        many_zeros[:50] = 0  # 50 zeros = 5%

        # Non-numeric column
        categorical = ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3)

        df = pd.DataFrame(
            {
                "few_zeros": few_zeros,
                "many_zeros": many_zeros,
                "categorical": categorical,
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["few_zeros", "many_zeros", "categorical"],
            __log=False,
        )

    def test_too_many_zeros_default_threshold(self):
        # Test with default threshold (3%)
        results, passed = TooManyZeroValues(self.vm_dataset)

        # Check return types
        self.assertIsInstance(results, list)
        self.assertIsInstance(passed, bool)

        # Check results structure
        self.assertEqual(len(results), 2)  # Should only check numeric columns
        for result in results:
            self.assertIn("Column", result)
            self.assertIn("Number of Zero Values", result)
            self.assertIn("Percentage of Zero Values (%)", result)
            self.assertIn("Pass/Fail", result)

        # Verify specific results
        few_zeros_result = next(r for r in results if r["Column"] == "few_zeros")
        many_zeros_result = next(r for r in results if r["Column"] == "many_zeros")

        self.assertEqual(few_zeros_result["Pass/Fail"], "Pass")  # 1% should pass
        self.assertEqual(many_zeros_result["Pass/Fail"], "Fail")  # 5% should fail
        self.assertFalse(passed)  # Overall test should fail due to many_zeros

    def test_custom_threshold(self):
        # Test with higher threshold (6%)
        results, passed = TooManyZeroValues(self.vm_dataset, max_percent_threshold=0.06)

        # Both columns should pass with higher threshold
        self.assertTrue(passed)
        self.assertTrue(all(row["Pass/Fail"] == "Pass" for row in results))
