import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.RunsTest import RunsTest


class TestRunsTest(unittest.TestCase):
    def test_returns_dataframe_and_raw_data(self):
        # Create a simple dataset with numeric columns
        df = pd.DataFrame(
            {
                "num1": [1, 2, 1, 2, 1, 2, 1, 2],  # Alternating pattern
                "num2": [1, 1, 1, 2, 2, 2, 1, 1],  # Runs pattern
                "num3": [1, 2, 3, 4, 5, 6, 7, 8],  # Sequential pattern
                "cat1": [
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                    "B",
                ],  # This should be ignored as it's not numeric
            }
        )

        vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=df,
            __log=False,
        )

        # Run the function
        result, raw_data = RunsTest(vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = ["feature", "stat", "pvalue"]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each numeric feature)
        self.assertEqual(len(result), len(vm_dataset.feature_columns_numeric))

        # Check if raw_data is instance of RawData
        self.assertIsInstance(raw_data, vm.RawData)
