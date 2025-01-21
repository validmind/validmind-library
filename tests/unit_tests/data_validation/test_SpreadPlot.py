import unittest
import pandas as pd
import matplotlib.pyplot as plt

import validmind as vm
from validmind import RawData

from validmind.errors import SkipTestError
from validmind.tests.data_validation.SpreadPlot import SpreadPlot


class TestSpreadPlot(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"A": range(100), "B": [i * 2 for i in range(100)]}, index=dates
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, feature_columns=["A", "B"], __log=False
        )

        # Create a dataset without datetime index
        df_no_datetime = pd.DataFrame(
            {"A": range(100), "B": [i * 2 for i in range(100)]}
        )

        self.vm_dataset_no_datetime = vm.init_dataset(
            input_id="test_dataset_no_datetime",
            dataset=df_no_datetime,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_spread_plot(self):
        result = SpreadPlot(self.vm_dataset)

        # The last item should be an instance of RawData
        self.assertIsInstance(result[-1], RawData)

        # Collect all figures except the last item
        figures = result[:-1]

        # Check that we get the correct number of figures (one per feature pair)
        self.assertEqual(len(figures), 1)  # Only one pair (A-B) for two features

        # Check that outputs are matplotlib figures
        for fig in figures:
            self.assertIsInstance(fig, plt.Figure)

        # Clean up
        plt.close("all")

    def test_no_datetime_index(self):
        # Should raise an error for non-datetime index
        with self.assertRaises(SkipTestError):
            SpreadPlot(self.vm_dataset_no_datetime)

        # Clean up
        plt.close("all")
