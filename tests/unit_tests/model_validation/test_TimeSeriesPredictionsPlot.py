import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import validmind as vm
from validmind.tests.model_validation.TimeSeriesPredictionsPlot import (
    TimeSeriesPredictionsPlot,
)


class TestTimeSeriesPredictionsPlot(unittest.TestCase):
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

    def test_return_type(self):
        """Test if function returns a Plotly figure."""
        result = TimeSeriesPredictionsPlot(self.vm_dataset, self.vm_model)
        self.assertIsInstance(result, go.Figure)

    def test_figure_properties(self):
        """Test if figure has expected properties."""
        fig = TimeSeriesPredictionsPlot(self.vm_dataset, self.vm_model)

        # Check if figure has exactly two traces (Actual and Predicted)
        self.assertEqual(len(fig.data), 2)

    def test_data_length(self):
        """Test if the plotted data has correct length."""
        fig = TimeSeriesPredictionsPlot(self.vm_dataset, self.vm_model)

        # Both traces should have same length as input data
        self.assertEqual(len(fig.data[0].x), len(self.df))
        self.assertEqual(len(fig.data[0].y), len(self.df))
        self.assertEqual(len(fig.data[1].x), len(self.df))
        self.assertEqual(len(fig.data[1].y), len(self.df))

    def test_datetime_index(self):
        """Test if x-axis uses datetime values."""
        fig = TimeSeriesPredictionsPlot(self.vm_dataset, self.vm_model)

        # Check if x values are datetime objects
        self.assertTrue(
            all(isinstance(x, (pd.Timestamp, datetime)) for x in fig.data[0].x)
        )
        self.assertTrue(
            all(isinstance(x, (pd.Timestamp, datetime)) for x in fig.data[1].x)
        )
