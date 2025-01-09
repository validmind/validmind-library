import unittest
import pandas as pd
import numpy as np
import statsmodels.api as sm
import validmind as vm
import plotly.graph_objects as go
from validmind import RawData
from validmind.tests.model_validation.statsmodels.RegressionCoeffs import (
    RegressionCoeffs,
)


class TestRegressionCoeffs(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset for regression
        np.random.seed(42)
        n_samples = 100

        # Create features with known relationships
        X = np.random.normal(0, 1, (n_samples, 3))

        # Create target with known coefficients
        true_coeffs = [2.0, -1.5, 0.5]
        y = (X * true_coeffs).sum(axis=1) + np.random.normal(0, 0.1, n_samples)

        # Create DataFrame first
        self.df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])

        # Add constant using statsmodels on the DataFrame
        X_with_const = sm.add_constant(self.df)
        self.df["const"] = 1.0  # Add constant to original df
        self.df["target"] = y

        # Create and train a statsmodels regression model
        self.model = sm.OLS(y, X_with_const).fit()

        # Wrap model in ValidMind model object
        self.vm_model = vm.init_model(
            input_id="statsmodel",
            model=self.model,
            __log=False,
        )

    def test_returns_tuple(self):
        # Run the function
        result = RegressionCoeffs(self.vm_model)

        # Check if result is a tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check if first element is a Plotly Figure
        self.assertIsInstance(result[0], go.Figure)

        # Check if second element is an instance of RawData
        self.assertIsInstance(result[1], RawData)

    def test_plot_properties(self):
        # Run the function
        fig, _ = RegressionCoeffs(self.vm_model)

        # Check if there is exactly one trace (bar plot)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "bar")

        # Check if error bars are present
        self.assertTrue(hasattr(fig.data[0], "error_y"))
        self.assertTrue(fig.data[0].error_y.visible)
