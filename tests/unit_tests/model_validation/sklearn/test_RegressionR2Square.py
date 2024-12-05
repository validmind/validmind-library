import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import validmind as vm
from validmind.tests.model_validation.sklearn.RegressionR2Square import (
    RegressionR2Square,
)


class TestRegressionR2Square(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with known relationships
        np.random.seed(42)
        n_samples = 100

        # Create features with different importance levels
        X = np.random.normal(0, 1, (n_samples, 2))

        # Create target with known relationship plus some noise
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)

        # Create DataFrame with features and target
        self.df = pd.DataFrame({"feature1": X[:, 0], "feature2": X[:, 1], "target": y})

        # Create and train a simple model
        X = self.df[["feature1", "feature2"]]
        y = self.df["target"]
        self.model = LinearRegression()
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

    def test_returns_dataframe(self):
        # Run the function
        result = RegressionR2Square(self.vm_dataset, self.vm_model)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = ["R-squared (R2) Score", "Adjusted R-squared (R2) Score"]
        self.assertTrue(all(col in result.columns for col in expected_columns))

        # Check if DataFrame has exactly one row
        self.assertEqual(len(result), 1)

    def test_perfect_prediction(self):
        # Create a perfect prediction scenario
        perfect_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [0.5, 1.5, 2.5],
                "target": [2, 4, 6],  # perfect linear relationship with feature1
            }
        )

        # Create and train a model that will give perfect predictions
        perfect_model = LinearRegression()
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

        # Calculate R2 scores
        result = RegressionR2Square(vm_perfect_dataset, vm_perfect_model)

        # Both R2 and Adjusted R2 should be very close to 1 for perfect predictions
        self.assertAlmostEqual(result["R-squared (R2) Score"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(
            result["Adjusted R-squared (R2) Score"].iloc[0], 1.0, places=5
        )

    def test_r2_score_range(self):
        result = RegressionR2Square(self.vm_dataset, self.vm_model)

        # R2 score should be between 0 and 1 for a reasonable model
        r2_score = result["R-squared (R2) Score"].iloc[0]
        adj_r2_score = result["Adjusted R-squared (R2) Score"].iloc[0]

        self.assertGreaterEqual(r2_score, 0)
        self.assertLessEqual(r2_score, 1)
        self.assertLessEqual(adj_r2_score, r2_score)  # Adjusted R2 should be <= R2

    def test_poor_prediction(self):
        # Create a poor prediction scenario (random predictions)
        poor_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [0.5, 1.5, 2.5],
                "target": [10, 2, 8],  # no clear relationship with features
            }
        )

        # Create a model with deliberately poor predictions
        poor_model = LinearRegression()
        poor_model.coef_ = np.array([0, 0])  # Set coefficients to 0
        poor_model.intercept_ = np.mean(poor_df["target"])  # Just predict mean

        # Initialize ValidMind objects
        vm_poor_dataset = vm.init_dataset(
            input_id="poor_dataset",
            dataset=poor_df,
            target_column="target",
            __log=False,
        )

        vm_poor_model = vm.init_model(
            input_id="poor_model",
            model=poor_model,
            __log=False,
        )

        # Assign predictions to the dataset
        vm_poor_dataset.assign_predictions(vm_poor_model)

        # Calculate R2 scores
        result = RegressionR2Square(vm_poor_dataset, vm_poor_model)

        # R2 score should be close to 0 for poor predictions
        self.assertLess(result["R-squared (R2) Score"].iloc[0], 0.1)
        self.assertLess(result["Adjusted R-squared (R2) Score"].iloc[0], 0.1)
