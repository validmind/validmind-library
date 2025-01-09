import unittest
import pandas as pd
import validmind as vm
import matplotlib.pyplot as plt
from validmind.tests.data_validation.RollingStatsPlot import RollingStatsPlot
from validmind import RawData


class TestRollingStatsPlot(unittest.TestCase):
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

    def test_rolling_stats_plot(self):
        outputs = RollingStatsPlot(self.vm_dataset, window_size=10)

        # Check that we get the correct number of figures (one per feature) plus raw data
        self.assertEqual(len(outputs), 3)

        # Check that first outputs are matplotlib figures
        for fig in outputs[:-1]:
            self.assertIsInstance(fig, plt.Figure)

        # Check that the last output is raw data
        self.assertIsInstance(outputs[-1], RawData)

        # Clean up
        plt.close("all")

    def test_no_datetime_index(self):
        # Should raise an error for non-datetime index
        with self.assertRaises(Exception) as context:
            RollingStatsPlot(self.vm_dataset_no_datetime)

        # Verify error message mentions datetime requirement
        self.assertIn("datetime", str(context.exception).lower())
