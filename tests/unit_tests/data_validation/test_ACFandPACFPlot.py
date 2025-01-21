import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.ACFandPACFPlot import ACFandPACFPlot
from plotly.graph_objects import Figure
from validmind import RawData


class TestACFandPACFPlot(unittest.TestCase):
    def setUp(self):
        # Create a simple time series dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame(
            {
                "value1": range(100),  # Simple linear trend
                "value2": [i % 7 for i in range(100)],  # Weekly seasonal pattern
            },
            index=dates,
        )

        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            __log=False,
        )

    def test_returns_expected_figures(self):
        # Run the function
        result = ACFandPACFPlot(self.vm_dataset)

        # Should return 6 items (ACF and PACF for each column + raw data)
        self.assertEqual(len(result), 5)

        # Check if the first 4 elements are Plotly figures
        for figure in result[:-1]:
            self.assertIsInstance(figure, Figure)

        # Check last element is of type RawData
        self.assertIsInstance(result[-1], RawData)

    def test_raises_error_for_non_datetime_index(self):
        # Create dataset with non-datetime index
        df_wrong_index = self.df.reset_index()
        vm_dataset_wrong_index = vm.init_dataset(
            input_id="dataset",
            dataset=df_wrong_index,
            __log=False,
        )

        # Check if it raises ValueError
        with self.assertRaises(ValueError) as context:
            ACFandPACFPlot(vm_dataset_wrong_index)

        self.assertEqual(str(context.exception), "Index must be a datetime type")

    def test_handles_nan_values(self):
        # Create dataset with some NaN values
        df_with_nans = self.df.copy()
        df_with_nans.iloc[0:2, 0] = None

        vm_dataset_with_nans = vm.init_dataset(
            input_id="dataset",
            dataset=df_with_nans,
            __log=False,
        )

        # Should run without errors
        result = ACFandPACFPlot(vm_dataset_with_nans)

        # Should still return 6 items (ACF and PACF per column + raw data)
        self.assertEqual(len(result), 5)

        # Check if last element is RawData
        self.assertIsInstance(result[-1], RawData)
