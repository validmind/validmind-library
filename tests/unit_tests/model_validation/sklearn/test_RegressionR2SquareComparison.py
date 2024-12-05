import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import validmind as vm
from validmind.tests.model_validation.sklearn.RegressionR2SquareComparison import (
    RegressionR2SquareComparison,
)


class TestRegressionR2SquareComparison(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create two sample datasets with known relationships
        np.random.seed(42)
        n_samples = 100

        # Create features and targets for first dataset (linear relationship)
        X1 = np.random.normal(0, 1, (n_samples, 2))
        y1 = 2 * X1[:, 0] + 0.5 * X1[:, 1] + np.random.normal(0, 0.1, n_samples)

        # Create features and targets for second dataset (non-linear relationship)
        # Using a simpler non-linear relationship with less noise
        X2 = np.random.uniform(
            -2, 2, (n_samples, 2)
        )  # Uniform distribution for better coverage
        y2 = (
            X2[:, 0] ** 2 + 0.5 * X2[:, 1] ** 2 + np.random.normal(0, 0.1, n_samples)
        )  # Quadratic relationship with less noise

        # Create DataFrames
        self.df1 = pd.DataFrame(
            {"feature1": X1[:, 0], "feature2": X1[:, 1], "target": y1}
        )

        self.df2 = pd.DataFrame(
            {"feature1": X2[:, 0], "feature2": X2[:, 1], "target": y2}
        )

        # Create and train two different models
        self.model1 = LinearRegression()
        self.model2 = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train both models on both datasets
        X1 = self.df1[["feature1", "feature2"]]
        y1 = self.df1["target"]
        X2 = self.df2[["feature1", "feature2"]]
        y2 = self.df2["target"]

        # Train models on their respective datasets
        self.model1.fit(X2, y2)  # Train linear model on non-linear data
        self.model2.fit(X2, y2)  # Train RF model on non-linear data

        # Initialize ValidMind datasets
        self.vm_dataset1 = vm.init_dataset(
            input_id="dataset1",
            dataset=self.df1,
            target_column="target",
            __log=False,
        )

        self.vm_dataset2 = vm.init_dataset(
            input_id="dataset2",
            dataset=self.df2,
            target_column="target",
            __log=False,
        )

        # Wrap models in ValidMind model objects
        self.vm_model1 = vm.init_model(
            input_id="linear_model",
            model=self.model1,
            __log=False,
        )

        self.vm_model2 = vm.init_model(
            input_id="rf_model",
            model=self.model2,
            __log=False,
        )

        # Assign predictions to datasets
        self.vm_dataset1.assign_predictions(self.vm_model1)
        self.vm_dataset1.assign_predictions(self.vm_model2)
        self.vm_dataset2.assign_predictions(self.vm_model1)
        self.vm_dataset2.assign_predictions(self.vm_model2)

    def test_returns_dataframe(self):
        # Run the function
        result = RegressionR2SquareComparison(
            [self.vm_dataset1, self.vm_dataset2], [self.vm_model1, self.vm_model2]
        )

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = ["Model", "Dataset", "R-Squared", "Adjusted R-Squared"]
        self.assertTrue(all(col in result.columns for col in expected_columns))

        # Check if DataFrame has correct number of rows (2 datasets * 2 models)
        self.assertEqual(len(result), 2)

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
        result = RegressionR2SquareComparison([vm_perfect_dataset], [vm_perfect_model])

        # Both R2 and Adjusted R2 should be very close to 1 for perfect predictions
        self.assertAlmostEqual(result["R-Squared"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(result["Adjusted R-Squared"].iloc[0], 1.0, places=5)

    def test_model_comparison(self):
        # Compare linear model vs random forest on non-linear dataset
        result = RegressionR2SquareComparison(
            [self.vm_dataset2, self.vm_dataset2], [self.vm_model1, self.vm_model2]
        )

        # Get R2 scores for both models
        linear_r2 = result[result["Model"] == "linear_model"]["R-Squared"].iloc[0]
        rf_r2 = result[result["Model"] == "rf_model"]["R-Squared"].iloc[0]

        # Random Forest should perform better on non-linear data
        self.assertGreater(rf_r2, linear_r2)

    def test_poor_prediction(self):
        # Create a poor prediction scenario
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
        result = RegressionR2SquareComparison([vm_poor_dataset], [vm_poor_model])

        # R2 scores should be close to 0 for poor predictions
        self.assertLess(result["R-Squared"].iloc[0], 0.1)
        self.assertLess(result["Adjusted R-Squared"].iloc[0], 0.1)
