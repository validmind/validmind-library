import unittest
import pandas as pd
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.PearsonCorrelationMatrix import (
    PearsonCorrelationMatrix,
)


class TestPearsonCorrelationMatrix(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with numeric columns
        self.df = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [2, 4, 6, 8, 10],  # Perfect correlation with num1
                "num3": [5, 4, 3, 2, 1],  # Perfect negative correlation with num1
                "cat1": [
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                ],  # This should be ignored as it's not numeric
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

    def test_returns_plotly_figure_and_raw_data(self):
        # Run the function
        result, raw_data = PearsonCorrelationMatrix(self.vm_dataset)

        # Check if result is a Plotly Figure
        self.assertIsInstance(result, go.Figure)

        # Check if the figure has data (should have one heatmap trace)
        self.assertEqual(len(result.data), 1)
        self.assertIsInstance(result.data[0], go.Heatmap)

        # Check if the heatmap has the correct dimensions (3x3 for numeric columns)
        self.assertEqual(len(result.data[0].x), 3)  # Number of numeric columns
        self.assertEqual(len(result.data[0].y), 3)  # Number of numeric columns

        # Check if raw_data is an instance of RawData
        self.assertIsInstance(raw_data, vm.RawData)
