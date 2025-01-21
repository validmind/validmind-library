import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.FeatureTargetCorrelationPlot import (
    FeatureTargetCorrelationPlot,
)


class TestFeatureTargetCorrelationPlot(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with numeric columns
        self.df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "feature3": [1, 3, 5, 7, 9],
                "target": [0, 1, 1, 0, 1],
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            __log=False,
        )

    def test_returns_plotly_figure_and_raw_data(self):
        # Run the function
        result = FeatureTargetCorrelationPlot(self.vm_dataset)

        # Check if result is a tuple of Plotly Figure and RawData
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        fig, raw_data = result

        # Check if fig is a Plotly Figure
        self.assertIsInstance(fig, go.Figure)

        # Check if the figure has data (at least one trace)
        self.assertTrue(len(fig.data) > 0)

        # Check if raw_data is an instance of RawData
        self.assertIsInstance(raw_data, vm.RawData)
