import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.BivariateScatterPlots import BivariateScatterPlots


class TestBivariateScatterPlots(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple dataset with numeric columns
        self.df = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [2, 4, 6, 8, 10], "col3": [1, 3, 5, 7, 9]}
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column=None,
            __log=False,
        )

    def test_returns_tuple_of_figures(self):
        # Run the function
        result = BivariateScatterPlots(self.vm_dataset)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Check if the tuple contains at least one figure (since we have multiple numeric columns)
        self.assertTrue(len(result) > 0)
