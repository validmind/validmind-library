import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TimeSeriesMissingValues import (
    TimeSeriesMissingValues,
)


class TestTimeSeriesMissingValues(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset with missing values
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "A": range(100),
                "B": [
                    i * 2 if i % 10 != 0 else None for i in range(100)
                ],  # Some missing values
            },
            index=dates,
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

    def test_time_series_missing_values(self):
        results, barplot, heatmap, passed = TimeSeriesMissingValues(self.vm_dataset)

        # Check return types
        self.assertIsInstance(results, list)
        self.assertIsInstance(barplot, go.Figure)
        self.assertIsInstance(heatmap, go.Figure)
        self.assertIsInstance(passed, bool)

        # Check results structure
        self.assertEqual(len(results), 2)  # One entry per feature
        for result in results:
            self.assertIn("Column", result)
            self.assertIn("Number of Missing Values", result)
            self.assertIn("Percentage of Missing Values (%)", result)
            self.assertIn("Pass/Fail", result)

        # Check specific values
        column_a = next(r for r in results if r["Column"] == "A")
        column_b = next(r for r in results if r["Column"] == "B")
        self.assertEqual(column_a["Number of Missing Values"], 0)
        self.assertEqual(
            column_b["Number of Missing Values"], 10
        )  # Every 10th value is None

    def test_no_datetime_index(self):
        # Should raise SkipTestError when no datetime index present
        with self.assertRaises(SkipTestError):
            TimeSeriesMissingValues(self.vm_dataset_no_datetime)
