import unittest
import pandas as pd
import numpy as np
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TimeSeriesOutliers import TimeSeriesOutliers


class TestTimeSeriesOutliers(unittest.TestCase):
    def setUp(self):
        # Create a sample time series dataset with known outliers
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Normal data with inserted outliers
        data_a = np.random.normal(0, 1, 100)
        data_a[50] = 10  # Insert obvious outlier

        data_b = np.random.normal(0, 1, 100)
        data_b[75] = -8  # Insert obvious outlier

        df = pd.DataFrame({"A": data_a, "B": data_b}, index=dates)

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, feature_columns=["A", "B"], __log=False
        )

        # Create dataset without datetime index
        df_no_datetime = pd.DataFrame({"A": data_a, "B": data_b})

        self.vm_dataset_no_datetime = vm.init_dataset(
            input_id="test_dataset_no_datetime",
            dataset=df_no_datetime,
            feature_columns=["A", "B"],
            __log=False,
        )

    def test_time_series_outliers(self):
        outlier_df, figures, passed = TimeSeriesOutliers(self.vm_dataset)

        # Check return types
        self.assertIsInstance(outlier_df, pd.DataFrame)
        self.assertIsInstance(figures, list)
        self.assertIsInstance(passed, bool)

        # Check that we have the expected number of figures (one per feature)
        self.assertEqual(len(figures), 2)

        # Check that outputs are plotly figures
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)

        # Check that outliers were detected
        self.assertGreater(len(outlier_df), 0)  # Should have found at least 2 outliers
        self.assertFalse(passed)  # Should fail due to outliers

        # Check outlier_df structure
        expected_columns = ["Column", "Z-Score", "Threshold", "Date", "Pass/Fail"]
        self.assertTrue(all(col in outlier_df.columns for col in expected_columns))

    def test_no_datetime_index(self):
        # Should raise SkipTestError when no datetime index present
        with self.assertRaises(SkipTestError):
            TimeSeriesOutliers(self.vm_dataset_no_datetime)
