import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import validmind as vm
import plotly.graph_objects as go
from validmind.tests.model_validation.statsmodels.PredictionProbabilitiesHistogram import (
    PredictionProbabilitiesHistogram,
)


class TestPredictionProbabilitiesHistogram(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with binary classification
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

        # Create DataFrame
        self.df = pd.DataFrame({"feature1": X[:, 0], "feature2": X[:, 1], "target": y})

        # Create and train a logistic regression model
        X = self.df[["feature1", "feature2"]]
        y = self.df["target"]
        self.model = LogisticRegression()
        self.model.fit(X, y)

        # Initialize ValidMind dataset
        self.vm_dataset = vm.init_dataset(
            input_id="dataset",
            dataset=self.df,
            target_column="target",
            __log=False,
        )

        # Wrap model in ValidMind model object
        self.vm_model = vm.init_model(
            input_id="model",
            model=self.model,
            __log=False,
        )

        # Assign predictions to the dataset
        self.vm_dataset.assign_predictions(self.vm_model)

    def test_returns_figure_and_raw_data(self):
        # Run the function
        figure, raw_data = PredictionProbabilitiesHistogram(
            self.vm_dataset, self.vm_model
        )

        # Check if figure is a Plotly Figure
        self.assertIsInstance(figure, go.Figure)

        # Check if raw_data is a RawData object
        self.assertIsInstance(raw_data, vm.RawData)

        # Check if figure has traces
        self.assertGreater(len(figure.data), 0)

    def test_perfect_separation(self):
        # Create a dataset with perfect class separation
        perfect_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, -1, -2, -3],
                "feature2": [1, 2, 3, -1, -2, -3],
                "target": [1, 1, 1, 0, 0, 0],
            }
        )

        # Create and train a model that will give perfect separation
        perfect_model = LogisticRegression()
        perfect_model.fit(perfect_df[["feature1", "feature2"]], perfect_df["target"])

        # Initialize ValidMind objects
        vm_perfect_dataset = vm.init_dataset(
            input_id="perfect_dataset",
            dataset=perfect_df,
            target_column="target",
            __log=False,
        )

        vm_perfect_model = vm.init_model(
            input_id="perfect_model",
            model=perfect_model,
            __log=False,
        )

        # Assign predictions to the dataset
        vm_perfect_dataset.assign_predictions(vm_perfect_model)

        # Generate histogram
        figure, raw_data = PredictionProbabilitiesHistogram(
            vm_perfect_dataset, vm_perfect_model
        )

        # Check if there are exactly two traces (one for each class)
        self.assertEqual(len(figure.data), 2)

    def test_probability_ranges(self):
        figure, raw_data = PredictionProbabilitiesHistogram(
            self.vm_dataset, self.vm_model
        )

        # Check if probabilities are within [0, 1] range
        for trace in figure.data:
            x_values = trace.x
            self.assertTrue(all(0 <= x <= 1 for x in x_values))
