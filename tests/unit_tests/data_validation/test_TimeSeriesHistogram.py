import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import validmind as vm
from validmind.tests.data_validation.TimeSeriesHistogram import TimeSeriesHistogram


class TestTimeSeriesHistogram(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample time series dataset with different distributions
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")

        np.random.seed(42)  # For reproducibility
        self.df = pd.DataFrame(
            {
                "normal_dist": np.random.normal(loc=0, scale=1, size=31),
                "uniform_dist": np.random.uniform(low=-3, high=3, size=31),
                "skewed_dist": np.exp(
                    np.random.normal(size=31)
                ),  # Log-normal distribution
                "missing_values": [1, 2, np.nan, 4, 5] * 6
                + [1],  # Including missing values
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

    def test_returns_tuple_of_figures_and_raw_data(self):
        # Run the function
        result = TimeSeriesHistogram(self.vm_dataset)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)

        # Should have one histogram per column
        self.assertEqual(len(result), len(self.df.columns))

        for fig in result:
            self.assertIsInstance(fig, go.Figure)

    def test_histogram_properties(self):
        result = TimeSeriesHistogram(self.vm_dataset)

        # Check the first histogram
        first_hist = result[0]

        # Should have two traces (histogram and violin plot)
        self.assertEqual(len(first_hist.data), 2)

        # First trace should be the histogram
        self.assertEqual(first_hist.data[0].type, "histogram")

        # Second trace should be the violin plot
        self.assertEqual(first_hist.data[1].type, "violin")

        # Should have a title
        self.assertIsNotNone(first_hist.layout.title)

    def test_handles_missing_values(self):
        result = TimeSeriesHistogram(self.vm_dataset)

        # Find the histogram for the 'missing_values' column
        missing_values_hist = next(
            fig for fig in result if "missing_values" in fig.layout.title.text
        )

        # The histogram should exist and have data
        self.assertIsNotNone(missing_values_hist)
        self.assertTrue(
            len(missing_values_hist.data[0].x) < 31
        )  # Less than total due to missing values

    def test_raises_error_for_non_datetime_index(self):
        # Check if ValueError is raised for non-datetime index
        with self.assertRaises(ValueError):
            TimeSeriesHistogram(self.invalid_vm_dataset)
