import unittest
import numpy as np
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.ShapiroWilk import ShapiroWilk


class TestShapiroWilk(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple dataset with numeric columns
        np.random.seed(42)  # For reproducibility
        self.df = pd.DataFrame(
            {
                "normal_dist": np.random.normal(0, 1, 100),  # Normal distribution
                "uniform_dist": np.random.uniform(0, 1, 100),  # Uniform distribution
                "exponential_dist": np.random.exponential(
                    1.0, 100
                ),  # Exponential distribution
                "categorical": ["A", "B"]
                * 50,  # This should be ignored as it's not numeric
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

    def test_returns_dataframe_with_expected_shape(self):
        # Run the function
        result = ShapiroWilk(self.vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = ["column", "stat", "pvalue"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each numeric feature)
        self.assertEqual(len(result), len(self.vm_dataset.feature_columns_numeric))

    def test_handles_different_distributions(self):
        # Run the function
        result = ShapiroWilk(self.vm_dataset)

        # The normal distribution should have a higher p-value than the exponential distribution
        normal_pvalue = result[result["column"] == "normal_dist"]["pvalue"].iloc[0]
        exp_pvalue = result[result["column"] == "exponential_dist"]["pvalue"].iloc[0]

        self.assertGreater(normal_pvalue, exp_pvalue)
