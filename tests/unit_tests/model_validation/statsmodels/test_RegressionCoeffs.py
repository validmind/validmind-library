import unittest
import pandas as pd
import numpy as np
import statsmodels.api as sm
import validmind as vm
import plotly.graph_objects as go
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

        # Check if second element is a DataFrame
        self.assertIsInstance(result[1], pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = [
            "Feature",
            "coef",
            "std err",
            "t",
            "P>|t|",
            "[0.025",
            "0.975]",
        ]
        self.assertTrue(all(col in result[1].columns for col in expected_columns))

    def test_coefficient_values(self):
        # Run the function
        _, coeffs_df = RegressionCoeffs(self.vm_model)

        # Check if coefficients are close to true values
        true_coeffs = {"const": 0.0, "feature1": 2.0, "feature2": -1.5, "feature3": 0.5}

        # First, verify all expected features are present
        features = coeffs_df["Feature"].values
        for feature in true_coeffs.keys():
            self.assertIn(
                feature, features, f"Feature {feature} not found in coefficients"
            )

        # Then check coefficient values
        for feature, true_coef in true_coeffs.items():
            mask = coeffs_df["Feature"] == feature
            self.assertTrue(any(mask), f"Feature {feature} not found in coefficients")

            estimated_coef = float(coeffs_df.loc[mask, "coef"].iloc[0])
            # Allow for some estimation error
            self.assertAlmostEqual(
                estimated_coef,
                true_coef,
                places=1,
                msg=f"Coefficient for {feature} differs significantly from true value",
            )

    def test_confidence_intervals(self):
        # Run the function
        _, coeffs_df = RegressionCoeffs(self.vm_model)

        # Check if confidence intervals are properly calculated
        for _, row in coeffs_df.iterrows():
            coef = float(row["coef"])
            lower_ci = float(row["[0.025"])
            upper_ci = float(row["0.975]"])

            # Check if confidence interval contains coefficient
            self.assertLess(lower_ci, coef)
            self.assertGreater(upper_ci, coef)

            # Check if confidence interval width is reasonable
            ci_width = upper_ci - lower_ci
            self.assertGreater(ci_width, 0)  # Should be positive
            self.assertLess(ci_width, 10)  # Should not be too wide

    def test_plot_properties(self):
        # Run the function
        fig, _ = RegressionCoeffs(self.vm_model)

        # Check if there is exactly one trace (bar plot)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "bar")

        # Check if error bars are present
        self.assertTrue(hasattr(fig.data[0], "error_y"))
        self.assertTrue(fig.data[0].error_y.visible)
