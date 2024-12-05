import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.TimeSeriesDescription import TimeSeriesDescription


class TestTimeSeriesDescription(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample time series dataset
        dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
        self.df = pd.DataFrame(
            {
                "value1": [
                    1,
                    2,
                    3,
                    np.nan,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                ],  # Including a missing value
                "value2": np.random.normal(0, 1, 10),
            },
            index=dates,
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

        # Create a non-datetime index dataset for testing error handling
        self.invalid_df = pd.DataFrame(
            {"value1": [1, 2, 3, 4, 5], "value2": [6, 7, 8, 9, 10]}
        )

        self.invalid_vm_dataset = vm.init_dataset(
            input_id="invalid_dataset",
            dataset=self.invalid_df,
            __log=False,
        )

    def test_returns_dataframe_with_expected_columns(self):
        # Run the function
        result = TimeSeriesDescription(self.vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = [
            "Variable",
            "Start Date",
            "End Date",
            "Frequency",
            "Num of Missing Values",
            "Count",
            "Min Value",
            "Max Value",
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each variable)
        self.assertEqual(len(result), len(self.df.columns))

    def test_correct_date_range(self):
        result = TimeSeriesDescription(self.vm_dataset)

        # Check start and end dates
        self.assertEqual(result["Start Date"].iloc[0], "2023-01-01")
        self.assertEqual(result["End Date"].iloc[0], "2023-01-10")

    def test_missing_values_count(self):
        result = TimeSeriesDescription(self.vm_dataset)

        # Check missing values count for value1 (should be 1)
        value1_row = result[result["Variable"] == "value1"].iloc[0]
        self.assertEqual(value1_row["Num of Missing Values"], 1)
        self.assertEqual(value1_row["Count"], 9)  # Total - Missing

    def test_raises_error_for_non_datetime_index(self):
        # Check if ValueError is raised for non-datetime index
        with self.assertRaises(ValueError):
            TimeSeriesDescription(self.invalid_vm_dataset)
