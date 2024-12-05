import unittest
import pandas as pd
import numpy as np
import validmind as vm
import plotly.graph_objects as go
from validmind.tests.model_validation.statsmodels.ScorecardHistogram import (
    ScorecardHistogram,
)


class TestScorecardHistogram(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with binary classification and scores
        np.random.seed(42)
        n_samples = 100

        # Create features that separate classes well
        X = np.random.normal(0, 1, (n_samples, 2))

        # Create binary target with clear separation
        # Points in first quadrant are mostly class 1, others are mostly class 0
        y = np.where((X[:, 0] > 0) & (X[:, 1] > 0), 1, 0)

        # Add some noise to make it more realistic
        noise_idx = np.random.choice(
            n_samples, size=int(n_samples * 0.1), replace=False
        )
        y[noise_idx] = 1 - y[noise_idx]

        # Generate synthetic credit scores
        # Higher scores for class 0 (non-default), lower scores for class 1 (default)
        scores = np.where(
            y == 0,
            np.random.normal(700, 50, n_samples),  # Good scores
            np.random.normal(500, 50, n_samples),
        )  # Poor scores

        # Create DataFrame
        self.df = pd.DataFrame(
            {"feature1": X[:, 0], "feature2": X[:, 1], "target": y, "score": scores}
        )

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

    def test_returns_figure(self):
        # Run the function
        result = ScorecardHistogram(self.vm_dataset)

        # Check if result is a Plotly Figure
        self.assertIsInstance(result, go.Figure)

        # Check if figure has traces
        self.assertGreater(len(result.data), 0)

    def test_missing_score_column(self):
        # Create dataset without score column
        df_no_score = self.df.drop("score", axis=1)
        vm_dataset_no_score = vm.init_dataset(
            input_id="no_score_dataset",
            dataset=df_no_score,
            target_column="target",
            __log=False,
        )

        # Check if ValueError is raised
        with self.assertRaises(ValueError):
            ScorecardHistogram(vm_dataset_no_score)

    def test_histogram_properties(self):
        result = ScorecardHistogram(self.vm_dataset)

        # Should have two traces (one for each class)
        self.assertEqual(len(result.data), 2)

        for trace in result.data:
            # Check if trace type is histogram
            self.assertEqual(trace.type, "histogram")

            # Check if scores are within reasonable range
            x_values = trace.x
            self.assertTrue(all(300 <= x <= 900 for x in x_values))

    def test_class_separation(self):

        # Now test the visualization
        result = ScorecardHistogram(self.vm_dataset)

        # Get scores for each class from the traces
        class_0_scores = None
        class_1_scores = None

        for trace in result.data:
            if "target = 0" in trace.name:
                class_0_scores = np.array(trace.x)
            elif "target = 1" in trace.name:
                class_1_scores = np.array(trace.x)

        # Verify we found both classes
        self.assertIsNotNone(class_0_scores, "Could not find trace for class 0")
        self.assertIsNotNone(class_1_scores, "Could not find trace for class 1")

        # Calculate means
        mean_class_0 = np.mean(class_0_scores)
        mean_class_1 = np.mean(class_1_scores)

        # Class 0 (non-default) should have higher scores
        self.assertGreater(
            mean_class_0,
            mean_class_1,
            f"Expected class 0 mean ({mean_class_0}) to be greater "
            f"than class 1 mean ({mean_class_1})",
        )
