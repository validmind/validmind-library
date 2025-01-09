import unittest
import pandas as pd
import matplotlib.pyplot as plt
import validmind as vm
from validmind.tests.data_validation.ScatterPlot import ScatterPlot


class TestScatterPlot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple dataset with numeric columns
        self.df = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [2, 4, 6, 8, 10],
                "num3": [1, 3, 5, 7, 9],
                "cat1": [
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                ],  # This should be handled by seaborn's pairplot
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

    def test_returns_tuple_of_figures_and_raw_data(self):
        # Run the function
        figure = ScatterPlot(self.vm_dataset)

        # Check if the first element is a matplotlib Figure
        self.assertIsInstance(figure, plt.Figure)

        # Check if all figures are properly closed
        self.assertEqual(len(plt.get_fignums()), 0)
