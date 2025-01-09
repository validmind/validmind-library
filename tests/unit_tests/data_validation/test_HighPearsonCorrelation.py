import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.HighPearsonCorrelation import (
    HighPearsonCorrelation,
)


class TestHighPearsonCorrelation(unittest.TestCase):
    def setUp(self):
        # Create dataset with known correlations
        np.random.seed(42)
        n_samples = 100

        # Create perfectly correlated features
        x = np.random.randn(n_samples)
        perfect_corr = x  # correlation = 1.0

        # Create moderately correlated feature
        moderate_corr = x + np.random.randn(n_samples) * 0.5  # correlation ≈ 0.9

        # Create weakly correlated feature
        weak_corr = x + np.random.randn(n_samples) * 2  # correlation ≈ 0.4

        # Create uncorrelated feature
        uncorrelated = np.random.randn(n_samples)  # correlation ≈ 0

        # Create non-numeric column
        categorical = ["A", "B"] * (n_samples // 2)

        df = pd.DataFrame(
            {
                "base": x,
                "perfect": perfect_corr,
                "moderate": moderate_corr,
                "weak": weak_corr,
                "uncorrelated": uncorrelated,
                "categorical": categorical,
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_correlation_structure(self):
        results, all_passed, raw_data = HighPearsonCorrelation(self.vm_dataset)

        # Check basic structure
        self.assertIsInstance(results, list)
        self.assertIsInstance(all_passed, bool)
        self.assertIsInstance(raw_data, vm.RawData)

        # Check result structure
        for result in results:
            self.assertIn("Columns", result)
            self.assertIn("Coefficient", result)
            self.assertIn("Pass/Fail", result)

    def test_correlation_values(self):
        results, _, _ = HighPearsonCorrelation(self.vm_dataset, max_threshold=0.5)

        # First result should be the perfect correlation
        perfect_corr = results[0]
        self.assertIn("base", perfect_corr["Columns"])
        self.assertIn("perfect", perfect_corr["Columns"])
        self.assertAlmostEqual(abs(perfect_corr["Coefficient"]), 1.0, places=5)
        self.assertEqual(perfect_corr["Pass/Fail"], "Fail")

        # Check moderate correlation
        moderate_found = False
        for result in results:
            if "moderate" in result["Columns"] and "base" in result["Columns"]:
                self.assertGreater(abs(result["Coefficient"]), 0.8)
                self.assertEqual(result["Pass/Fail"], "Fail")
                moderate_found = True
                break
        self.assertTrue(moderate_found)

    def test_categorical_exclusion(self):
        results, _, _ = HighPearsonCorrelation(self.vm_dataset)

        # Verify categorical column is not in results
        for result in results:
            self.assertNotIn("categorical", result["Columns"])
