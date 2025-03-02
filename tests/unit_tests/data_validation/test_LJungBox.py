import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.LJungBox import LJungBox


class TestLJungBox(unittest.TestCase):
    def test_returns_dataframe_with_expected_shape(self):
        # Create a simple time series dataset
        df = pd.DataFrame(
            {
                "series1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "series2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "series3": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            }
        )

        vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=df,
            __log=False,
        )

        # Run the function
        result, raw_data = LJungBox(vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if raw_data is a RawData object
        self.assertIsInstance(raw_data, vm.RawData)

        # Check if the DataFrame has the expected columns
        expected_columns = ["column", "stat", "pvalue"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each column)
        self.assertEqual(len(result), len(df.columns))
