import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.TimeSeriesDescriptiveStatistics import (
    TimeSeriesDescriptiveStatistics,
)


class TestTimeSeriesDescriptiveStatistics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample time series dataset
        dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")

        # Create data with known statistics
        self.df = pd.DataFrame(
            {
                "normal_dist": np.random.normal(
                    loc=5, scale=1, size=10
                ),  # Normal distribution
                "skewed_dist": np.exp(
                    np.random.normal(size=10)
                ),  # Log-normal (skewed) distribution
                "missing_values": [
                    1,
                    2,
                    np.nan,
                    4,
                    5,
                    6,
                    7,
                    np.nan,
                    9,
                    10,
                ],  # Including missing values
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
        result = TimeSeriesDescriptiveStatistics(self.vm_dataset)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = [
            "Variable",
            "Start Date",
            "End Date",
            "Min",
            "Mean",
            "Max",
            "Skewness",
            "Kurtosis",
            "Count",
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (one for each variable)
        self.assertEqual(len(result), len(self.df.columns))

    def test_correct_date_range(self):
        result = TimeSeriesDescriptiveStatistics(self.vm_dataset)

        # Check start and end dates
        self.assertEqual(result["Start Date"].iloc[0], "2023-01-01")
        self.assertEqual(result["End Date"].iloc[0], "2023-01-10")

    def test_handles_missing_values(self):
        result = TimeSeriesDescriptiveStatistics(self.vm_dataset)

        # Check missing values handling for 'missing_values' column
        missing_values_row = result[result["Variable"] == "missing_values"].iloc[0]
        self.assertEqual(
            missing_values_row["Count"], 8
        )  # Should be 10 - 2 missing values

    def test_skewness_detection(self):
        result = TimeSeriesDescriptiveStatistics(self.vm_dataset)

        # The log-normal distribution should be more skewed than the normal distribution
        normal_skew = abs(
            result[result["Variable"] == "normal_dist"]["Skewness"].iloc[0]
        )
        skewed_skew = abs(
            result[result["Variable"] == "skewed_dist"]["Skewness"].iloc[0]
        )
        self.assertGreater(skewed_skew, normal_skew)

    def test_raises_error_for_non_datetime_index(self):
        # Check if ValueError is raised for non-datetime index
        with self.assertRaises(ValueError):
            TimeSeriesDescriptiveStatistics(self.invalid_vm_dataset)
