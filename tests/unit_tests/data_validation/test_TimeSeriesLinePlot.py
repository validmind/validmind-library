import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind import RawData
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TimeSeriesLinePlot import TimeSeriesLinePlot


class TestTimeSeriesLinePlot(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"A": range(100), "B": [i * 2 for i in range(100)]}, index=dates
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, feature_columns=["A", "B"], __log=False
        )

        # Create dataset without datetime index
        df_no_datetime = pd.DataFrame(
            {"A": range(100), "B": [i * 2 for i in range(100)]}
        )

        self.vm_dataset_no_datetime = vm.init_dataset(
            input_id="test_dataset_no_datetime",
            dataset=df_no_datetime,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_time_series_line_plot(self):
        figures = TimeSeriesLinePlot(self.vm_dataset)

        # Check that we get the correct number of figures plus raw data (one per feature + RawData)
        self.assertIsInstance(figures, tuple)
        self.assertEqual(
            len(figures), 3
        )  # Should have 2 figures for A and B and 1 RawData

        # Check that the first two outputs are plotly figures
        for fig in figures[:2]:
            self.assertIsInstance(fig, go.Figure)

        # Check that the last output is RawData
        self.assertIsInstance(figures[-1], RawData)

    def test_no_datetime_index(self):
        # Should raise SkipTestError when no datetime index present
        with self.assertRaises(SkipTestError):
            TimeSeriesLinePlot(self.vm_dataset_no_datetime)
