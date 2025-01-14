import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TabularDateTimeHistograms import (
    TabularDateTimeHistograms,
)


class TestTabularDateTimeHistograms(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with datetime index
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({"A": range(100), "B": range(100, 200)}, index=dates)

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, feature_columns=["A", "B"], __log=False
        )

        # Create dataset without datetime index
        df_no_datetime = pd.DataFrame({"A": range(100), "B": range(100, 200)})

        self.vm_dataset_no_datetime = vm.init_dataset(
            input_id="test_dataset_no_datetime",
            dataset=df_no_datetime,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_datetime_histograms(self):
        figure, raw_data = TabularDateTimeHistograms(self.vm_dataset)

        # Check that output is a plotly figure
        self.assertIsInstance(figure, go.Figure)

        # Check that raw data is an instance of RawData
        self.assertIsInstance(raw_data, vm.RawData)

    def test_no_datetime_index(self):
        # Should raise SkipTestError when no datetime index present
        with self.assertRaises(SkipTestError):
            TabularDateTimeHistograms(self.vm_dataset_no_datetime)
