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
        figure, raw_data = IsolationForestOutliers(self.vm_dataset, contamination=0.1)

        # Check return types
        self.assertIsInstance(figure, plt.Figure)
        self.assertIsInstance(raw_data, vm.RawData)

        # Check that the figure has at least one axes
        self.assertGreater(len(figure.axes), 0)

    def test_feature_columns_validation(self):
        # Test with valid feature columns
        try:
            figure, raw_data = IsolationForestOutliers(
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
        figure_low, raw_data_low = IsolationForestOutliers(
            self.vm_dataset, contamination=0.05
        )
        figure_high, raw_data_high = IsolationForestOutliers(
            self.vm_dataset, contamination=0.2
        )

        # Check that figures have at least one axes
        self.assertGreater(len(figure_low.axes), 0)
        self.assertGreater(len(figure_high.axes), 0)
