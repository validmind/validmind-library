import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.LaggedCorrelationHeatmap import (
    LaggedCorrelationHeatmap,
)
import plotly.graph_objects as go
from validmind import RawData


class TestLaggedCorrelationHeatmap(unittest.TestCase):
    def setUp(self):
        # Create test dataset with known correlations
        np.random.seed(42)
        n_samples = 100

        # Create datetime index
        date_rng = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

        # Create feature with known lag correlation
        x = np.random.normal(0, 1, n_samples)
        # Create target that's correlated with x shifted by 2 periods
        y = np.roll(x, 2) + np.random.normal(0, 0.1, n_samples)

        # Create another feature with no lag correlation
        z = np.random.normal(0, 1, n_samples)

        df = pd.DataFrame({"feature1": x, "feature2": z, "target": y}, index=date_rng)

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, target_column="target", __log=False
        )

    def test_heatmap_structure(self):
        fig, raw_data = LaggedCorrelationHeatmap(self.vm_dataset, num_lags=5)

        # Check return type for fig
        self.assertIsInstance(fig, go.Figure)

        # Check figure has data
        self.assertTrue(len(fig.data) > 0)

        # Check title contains target column name
        self.assertIn("target", fig.layout.title.text)

        # Check x-axis label
        self.assertEqual(fig.layout.xaxis.title.text, "Lags")

    def test_correlation_values(self):
        fig, _ = LaggedCorrelationHeatmap(self.vm_dataset, num_lags=5)

        # Get correlation values from heatmap
        heatmap_data = fig.data[0]

        # Check dimensions
        self.assertEqual(len(heatmap_data.z), 2)  # Two features
        self.assertEqual(len(heatmap_data.z[0]), 6)  # num_lags + 1

        # Check feature names
        self.assertEqual(heatmap_data.y[0], "feature1")
        self.assertEqual(heatmap_data.y[1], "feature2")

        # Check lag labels (now comparing as tuples)
        expected_labels = tuple(
            str(i) for i in range(6)
        )  # ('0', '1', '2', '3', '4', '5')
        self.assertEqual(heatmap_data.x, expected_labels)

    def test_num_lags_parameter(self):
        # Test with different number of lags
        fig_small, _ = LaggedCorrelationHeatmap(self.vm_dataset, num_lags=3)
        fig_large, _ = LaggedCorrelationHeatmap(self.vm_dataset, num_lags=8)

        # Check dimensions match num_lags parameter
        self.assertEqual(len(fig_small.data[0].x), 4)  # num_lags + 1
        self.assertEqual(len(fig_large.data[0].x), 9)  # num_lags + 1

    def test_correlation_pattern(self):
        fig, _ = LaggedCorrelationHeatmap(self.vm_dataset, num_lags=5)

        # Get correlation values for feature1
        correlations = fig.data[0].z[0]

        # Check that lag 2 has higher correlation than lag 0
        # (since we created target with 2-period lag)
        self.assertGreater(abs(correlations[2]), abs(correlations[0]))

    def test_raw_data_output(self):
        _, raw_data = LaggedCorrelationHeatmap(self.vm_dataset, num_lags=5)

        # Check that raw_data is instance of RawData
        self.assertIsInstance(raw_data, RawData)
