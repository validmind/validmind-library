import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.IQROutliersTable import IQROutliersTable


class TestIQROutliersTable(unittest.TestCase):
    def setUp(self):
        # Create a dataset with known outliers
        n_samples = 100

        # Create controlled "normal" data without outliers
        normal_data = np.array(
            [1.0] * 25 + [2.0] * 50 + [3.0] * 25
        )  # Values within IQR

        # Create data with outliers (same length as normal_data)
        data_with_outliers = normal_data.copy()
        # Replace some values with outliers
        outlier_indices = [0, 1, 2, 3]
        data_with_outliers[outlier_indices] = [10, 15, -10, -15]  # Clear outliers

        # Binary feature (same length as other columns)
        binary_data = np.zeros(n_samples)
        binary_data[:50] = 1  # Half 0s, half 1s

        df = pd.DataFrame(
            {
                "normal": normal_data,
                "with_outliers": data_with_outliers,
                "binary": binary_data,
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_outliers_structure(self):
        result, raw_data = IQROutliersTable(self.vm_dataset)

        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertIsInstance(raw_data, vm.RawData)
        self.assertIn("Summary of Outliers Detected by IQR Method", result)

        # Check result structure
        outliers_summary = result["Summary of Outliers Detected by IQR Method"]
        self.assertIsInstance(outliers_summary, list)

        for summary in outliers_summary:
            self.assertIn("Variable", summary)
            self.assertIn("Total Count of Outliers", summary)
            self.assertIn("Mean Value of Variable", summary)
            self.assertIn("Minimum Outlier Value", summary)
            self.assertIn("Outlier Value at 25th Percentile", summary)
            self.assertIn("Outlier Value at 50th Percentile", summary)
            self.assertIn("Outlier Value at 75th Percentile", summary)
            self.assertIn("Maximum Outlier Value", summary)

    def test_outliers_detection(self):
        result, raw_data = IQROutliersTable(self.vm_dataset)
        outliers_summary = result["Summary of Outliers Detected by IQR Method"]

        # Check that outliers are detected in the 'with_outliers' column
        with_outliers_summary = next(
            (s for s in outliers_summary if s["Variable"] == "with_outliers"), None
        )
        self.assertIsNotNone(with_outliers_summary)
        self.assertGreater(with_outliers_summary["Total Count of Outliers"], 0)

        # Check that no outliers are detected in the 'normal' column
        normal_summary = next(
            (s for s in outliers_summary if s["Variable"] == "normal"), None
        )
        self.assertIsNone(normal_summary)

    def test_binary_exclusion(self):
        result, raw_data = IQROutliersTable(self.vm_dataset)
        outliers_summary = result["Summary of Outliers Detected by IQR Method"]

        # Verify binary column is not in results
        for summary in outliers_summary:
            self.assertNotIn("binary", summary["Variable"])
