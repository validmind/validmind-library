import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.IsolationForestOutliers import (
    IsolationForestOutliers,
)
import matplotlib.pyplot as plt


class TestIsolationForestOutliers(unittest.TestCase):
    def setUp(self):
        # Create a dataset with known outliers
        np.random.seed(42)
        n_samples = 100

        # Generate normal data
        normal_data = np.random.normal(loc=0, scale=1, size=(n_samples, 2))

        # Add outliers
        outliers = np.random.uniform(low=-10, high=10, size=(10, 2))
        data = np.vstack([normal_data, outliers])

        df = pd.DataFrame(data, columns=["feature1", "feature2"])

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_outliers_detection(self):
        figures = IsolationForestOutliers(self.vm_dataset, contamination=0.1)

        # Check return type
        self.assertIsInstance(figures, tuple)

        # Check that at least one figure is returned
        self.assertGreater(len(figures), 0)

        # Check each figure
        for fig in figures:
            self.assertIsInstance(fig, plt.Figure)

    def test_feature_columns_validation(self):
        # Test with valid feature columns
        try:
            IsolationForestOutliers(
                self.vm_dataset, feature_columns=["feature1", "feature2"]
            )
        except ValueError:
            self.fail("IsolationForestOutliers raised ValueError unexpectedly!")

        # Test with invalid feature columns
        with self.assertRaises(ValueError):
            IsolationForestOutliers(
                self.vm_dataset, feature_columns=["invalid_feature"]
            )

    def test_contamination_parameter(self):
        # Test with different contamination levels
        figures_low_contamination = IsolationForestOutliers(
            self.vm_dataset, contamination=0.05
        )
        figures_high_contamination = IsolationForestOutliers(
            self.vm_dataset, contamination=0.2
        )

        # Check that figures are returned for both contamination levels
        self.assertGreater(len(figures_low_contamination), 0)
        self.assertGreater(len(figures_high_contamination), 0)
