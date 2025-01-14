import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.TimeSeriesPredictionWithCI import (
    TimeSeriesPredictionWithCI,
)


class TestTimeSeriesPredictionWithCI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample time series data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create simple sine wave with some noise for actual values
        X = np.arange(100)
        y_true = np.sin(X * 0.1) + np.random.normal(0, 0.1, 100)

        # Create predictions with slight offset
        y_pred = np.sin(X * 0.1) + 0.1 + np.random.normal(0, 0.1, 100)

        # Create DataFrame with datetime index
        self.df = pd.DataFrame({"target": y_true, "predictions": y_pred}, index=dates)

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="timeseries_dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

        # Initialize ValidMind model
        self.vm_model = vm.init_model(
            input_id="timeseries_model",
            attributes={
                "architecture": "TimeSeriesModel",
                "language": "Python",
            },
            __log=False,
        )

        # Link predictions
        self.vm_dataset.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        fig, breaches_df, raw_data = TimeSeriesPredictionWithCI(
            self.vm_dataset, self.vm_model
        )

        # Check return types
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(breaches_df, pd.DataFrame)
        self.assertIsInstance(raw_data, RawData)

    def test_figure_properties(self):
        """Test if figure has expected properties."""
        fig, _, _ = TimeSeriesPredictionWithCI(self.vm_dataset, self.vm_model)

        # Check if figure has exactly four traces (Actual, Predicted, CI Lower, CI Upper)
        self.assertEqual(len(fig.data), 4)

    def test_breaches_dataframe(self):
        """Test if breaches DataFrame has expected structure and values."""
        _, breaches_df, _ = TimeSeriesPredictionWithCI(self.vm_dataset, self.vm_model)

        # Check columns
        expected_columns = [
            "Confidence Level",
            "Total Breaches",
            "Upper Breaches",
            "Lower Breaches",
        ]
        self.assertListEqual(list(breaches_df.columns), expected_columns)

        # Check that there's exactly one row
        self.assertEqual(len(breaches_df), 1)

        # Check confidence level is correct (default 0.95)
        self.assertEqual(breaches_df["Confidence Level"].iloc[0], 0.95)

        # Check breach counts are non-negative integers
        self.assertGreaterEqual(breaches_df["Total Breaches"].iloc[0], 0)
        self.assertGreaterEqual(breaches_df["Upper Breaches"].iloc[0], 0)
        self.assertGreaterEqual(breaches_df["Lower Breaches"].iloc[0], 0)

        # Check total breaches equals sum of upper and lower breaches
        self.assertEqual(
            breaches_df["Total Breaches"].iloc[0],
            breaches_df["Upper Breaches"].iloc[0]
            + breaches_df["Lower Breaches"].iloc[0],
        )

    def test_custom_confidence(self):
        """Test if custom confidence level works."""
        custom_confidence = 0.90
        _, breaches_df, _ = TimeSeriesPredictionWithCI(
            self.vm_dataset, self.vm_model, confidence=custom_confidence
        )

        # Check if custom confidence level is used
        self.assertEqual(breaches_df["Confidence Level"].iloc[0], custom_confidence)

    def test_data_length(self):
        """Test if the plotted data has correct length."""
        fig, _, _ = TimeSeriesPredictionWithCI(self.vm_dataset, self.vm_model)

        # All traces should have same length as input data
        for trace in fig.data:
            self.assertEqual(len(trace.x), len(self.df))
            self.assertEqual(len(trace.y), len(self.df))

    def test_datetime_index(self):
        """Test if x-axis uses datetime values."""
        fig, _, _ = TimeSeriesPredictionWithCI(self.vm_dataset, self.vm_model)

        # Check if x values are datetime objects for all traces
        for trace in fig.data:
            self.assertTrue(
                all(isinstance(x, (pd.Timestamp, datetime)) for x in trace.x)
            )
