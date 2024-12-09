import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import validmind as vm
from validmind.tests.model_validation.TimeSeriesR2SquareBySegments import (
    TimeSeriesR2SquareBySegments,
)


class TestTimeSeriesR2SquareBySegments(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample time series data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create predictable pattern with some noise
        X = np.arange(100)
        y_true = 2 * X + np.random.normal(0, 1, 100)  # Linear pattern with noise

        # Create predictions with slight offset
        y_pred = 2 * X + 1 + np.random.normal(0, 1, 100)

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
            attributes={"architecture": "TimeSeriesModel", "language": "Python"},
            __log=False,
        )

        # Link predictions
        self.vm_dataset.assign_predictions(
            self.vm_model, prediction_column="predictions"
        )

    def test_return_types(self):
        """Test if function returns expected types."""
        fig, results_df = TimeSeriesR2SquareBySegments(self.vm_dataset, self.vm_model)

        # Check return types
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(results_df, pd.DataFrame)

    def test_results_dataframe(self):
        """Test if results DataFrame has expected structure."""
        _, results_df = TimeSeriesR2SquareBySegments(self.vm_dataset, self.vm_model)

        # Check columns
        expected_columns = ["Segments", "Start Date", "End Date", "R-Squared"]
        self.assertListEqual(list(results_df.columns), expected_columns)

        # Check number of segments (default is 2)
        self.assertEqual(len(results_df), 2)

        # Check segment names
        self.assertEqual(results_df["Segments"].iloc[0], "Segment 1")
        self.assertEqual(results_df["Segments"].iloc[1], "Segment 2")

        # Check R-squared values are within valid range
        self.assertTrue(all(score <= 1.0 for score in results_df["R-Squared"]))

    def test_custom_segments(self):
        """Test if custom segments work correctly."""
        dates = self.df.index
        custom_segments = {
            "start_date": [dates[0], dates[33], dates[66]],
            "end_date": [dates[32], dates[65], dates[-1]],
        }

        _, results_df = TimeSeriesR2SquareBySegments(
            self.vm_dataset, self.vm_model, segments=custom_segments
        )

        # Check number of segments matches custom segments
        self.assertEqual(len(results_df), 3)

        # Check segment names
        self.assertEqual(results_df["Segments"].iloc[0], "Segment 1")
        self.assertEqual(results_df["Segments"].iloc[1], "Segment 2")
        self.assertEqual(results_df["Segments"].iloc[2], "Segment 3")
