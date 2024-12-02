import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.IQROutliersBarPlot import IQROutliersBarPlot
import plotly.graph_objects as go


class TestIQROutliersBarPlot(unittest.TestCase):
    def setUp(self):
        # Create a dataset with known outliers
        n_samples = 100

        # Create controlled "normal" data without outliers
        normal_data = np.array(
            [1.0] * 25 + [2.0] * 50 + [3.0] * 25
        )  # Values within IQR

        # Create data with outliers (same length as normal_data)
        data_with_outliers = normal_data.copy()
        # Replace some values with outliers across different percentiles
        data_with_outliers[0:4] = [-15, -10, 10, 15]  # Clear outliers

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

    def test_plot_structure(self):
        figures = IQROutliersBarPlot(self.vm_dataset)

        # Check return type
        self.assertIsInstance(figures, tuple)

        # Check each figure
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)

            # Check figure components
            self.assertEqual(len(fig.data), 1)  # One bar trace
            self.assertIsInstance(fig.data[0], go.Bar)

    def test_plot_data(self):
        figures = IQROutliersBarPlot(self.vm_dataset)

        # Should have at least one figure (for the column with outliers)
        self.assertGreater(len(figures), 0)

        # Find the figure for 'with_outliers' column
        outliers_fig = next(
            (fig for fig in figures if fig.layout.title.text == "with_outliers"), None
        )
        self.assertIsNotNone(outliers_fig)

        # Check bar data
        bar_data = outliers_fig.data[0].y
        self.assertEqual(len(bar_data), 4)  # Should have 4 bars for percentile ranges
        self.assertGreater(sum(bar_data), 0)  # Should have some outliers

    def test_binary_exclusion(self):
        figures = IQROutliersBarPlot(self.vm_dataset)

        # Check that binary column is not included
        figure_titles = [fig.layout.title.text for fig in figures]
        self.assertNotIn("binary", figure_titles)
