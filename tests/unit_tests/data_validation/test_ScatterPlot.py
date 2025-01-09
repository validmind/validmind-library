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
        result = ScatterPlot(self.vm_dataset)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Check if the tuple contains exactly two elements
        self.assertEqual(len(result), 2)

        # Check if the first element is a matplotlib Figure
        self.assertIsInstance(result[0], plt.Figure)

        # Check if the second element is an instance of RawData
        self.assertIsInstance(result[1], vm.RawData)

        # Check if all figures are properly closed
        self.assertEqual(len(plt.get_fignums()), 0)
