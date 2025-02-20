import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.JarqueBera import JarqueBera


class TestJarqueBera(unittest.TestCase):
    def test_returns_dataframe_and_rawdata(self):
        # Create a simple dataset with numeric columns
        df = pd.DataFrame(
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
                ],  # This should be ignored as it's not numeric
            }
        )

        vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=df,
            __log=False,
        )

        # Run the function
        result, raw_data = JarqueBera(vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if raw_data is a RawData object
        self.assertIsInstance(raw_data, vm.RawData)

        # Check if the DataFrame has the expected columns
        expected_columns = ["column", "stat", "pvalue", "skew", "kurtosis"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each numeric feature)
        self.assertEqual(len(result), len(vm_dataset.feature_columns_numeric))
