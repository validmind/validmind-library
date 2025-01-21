import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TimeSeriesFrequency import TimeSeriesFrequency
from validmind import RawData


class TestTimeSeriesFrequency(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset with daily frequency
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

    def test_time_series_frequency(self):
        frequencies, figure, passed, raw_data = TimeSeriesFrequency(self.vm_dataset)

        # Check return types
        self.assertIsInstance(frequencies, list)
        self.assertIsInstance(figure, go.Figure)
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(raw_data, RawData)  # Check the new raw_data type

        # Check frequencies structure
        self.assertEqual(len(frequencies), 2)  # One entry per feature
        for freq in frequencies:
            self.assertIn("Variable", freq)
            self.assertIn("Frequency", freq)

        # Check that frequencies match (should be 'Daily' for both columns)
        self.assertTrue(all(f["Frequency"] == "Daily" for f in frequencies))
        self.assertTrue(passed)  # All frequencies should match

    def test_no_datetime_index(self):
        # Should raise SkipTestError when no datetime index present
        with self.assertRaises(SkipTestError):
            TimeSeriesFrequency(self.vm_dataset_no_datetime)
