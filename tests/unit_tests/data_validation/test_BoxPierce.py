import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.BoxPierce import BoxPierce


class TestBoxPierce(unittest.TestCase):
    def setUp(self):
        # Create a simple time series dataset
        self.df = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [2, 4, 6, 8, 10], "col3": [1, 3, 5, 7, 9]}
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column=None,
            __log=False,
        )

    def test_returns_dataframe_with_expected_columns(self):
        # Run the function
        result = BoxPierce(self.vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = ["column", "stat", "pvalue"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each input column)
        self.assertEqual(len(result), len(self.df.columns))
