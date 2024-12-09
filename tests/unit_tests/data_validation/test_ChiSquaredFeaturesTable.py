import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.ChiSquaredFeaturesTable import (
    ChiSquaredFeaturesTable,
)


class TestChiSquaredFeaturesTable(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with categorical columns
        self.df = pd.DataFrame(
            {
                "cat1": ["A", "B", "A", "B", "A"],
                "cat2": ["X", "X", "Y", "Y", "X"],
                "target": [0, 1, 1, 0, 1],
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

    def test_returns_dataframe_with_expected_shape(self):
        # Run the function
        result = ChiSquaredFeaturesTable(self.vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = [
            "Variable",
            "Chi-squared statistic",
            "p-value",
            "Threshold",
            "Pass/Fail",
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each categorical feature)
        self.assertEqual(len(result), len(self.vm_dataset.feature_columns_categorical))
