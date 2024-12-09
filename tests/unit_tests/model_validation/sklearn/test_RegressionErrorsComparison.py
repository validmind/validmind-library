import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import validmind as vm
from validmind.tests.model_validation.sklearn.RegressionErrorsComparison import (
    RegressionErrorsComparison,
)


class TestRegressionErrorsComparison(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create two sample datasets with known relationships
        np.random.seed(42)
        n_samples = 100

        # Create features and targets for first dataset
        X1 = np.random.normal(0, 1, (n_samples, 2))
        y1 = 2 * X1[:, 0] + 0.5 * X1[:, 1] + np.random.normal(0, 0.1, n_samples)

        # Create features and targets for second dataset (non-linear relationship)
        X2 = np.random.normal(0, 1, (n_samples, 2))
        y2 = (
            3 * np.square(X2[:, 0])
            - np.exp(X2[:, 1])
            + np.random.normal(0, 0.2, n_samples)
        )

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

        # Train models on first dataset
        X = self.df1[["feature1", "feature2"]]
        y = self.df1["target"]
        self.model1.fit(X, y)
        self.model2.fit(X, y)

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
        result = RegressionErrorsComparison(
            [self.vm_dataset1, self.vm_dataset2], [self.vm_model1, self.vm_model2]
        )

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = [
            "Model",
            "Dataset",
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Mean Absolute Percentage Error (MAPE)",
            "Mean Bias Deviation (MBD)",
        ]
        self.assertTrue(all(col in result.columns for col in expected_columns))

        # Check if DataFrame has correct number of rows (2 datasets * 2 models)
        self.assertEqual(len(result), 2)

    def test_error_metrics_range(self):
        result = RegressionErrorsComparison([self.vm_dataset1], [self.vm_model1])

        # All error metrics should be non-negative (except MBD)
        self.assertGreaterEqual(result["Mean Absolute Error (MAE)"].iloc[0], 0)
        self.assertGreaterEqual(result["Mean Squared Error (MSE)"].iloc[0], 0)
        self.assertGreaterEqual(
            result["Mean Absolute Percentage Error (MAPE)"].iloc[0], 0
        )

    def test_model_comparison(self):
        # Create a perfect prediction scenario
        X_perfect = np.array([[1], [2], [3]])
        y_perfect = 2 * X_perfect.ravel()

        perfect_df = pd.DataFrame({"feature1": X_perfect.ravel(), "target": y_perfect})

        # Create two models: one perfect, one imperfect
        perfect_model = LinearRegression()
        imperfect_model = LinearRegression()

        # Train perfect model correctly
        perfect_model.fit(X_perfect, y_perfect)

        # Train imperfect model on different data
        imperfect_model.coef_ = np.array([1.5])  # Deliberately wrong coefficient
        imperfect_model.intercept_ = 0.5  # Add some bias

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

        vm_imperfect_model = vm.init_model(
            input_id="imperfect_model",
            model=imperfect_model,
            __log=False,
        )

        # Assign predictions
        vm_perfect_dataset.assign_predictions(vm_perfect_model)
        vm_perfect_dataset.assign_predictions(vm_imperfect_model)

        # Compare models
        result = RegressionErrorsComparison(
            [vm_perfect_dataset, vm_perfect_dataset],
            [vm_perfect_model, vm_imperfect_model],
        )

        # Perfect model should have lower errors than imperfect model
        perfect_mae = result[result["Model"] == "perfect_model"][
            "Mean Absolute Error (MAE)"
        ].iloc[0]
        imperfect_mae = result[result["Model"] == "imperfect_model"][
            "Mean Absolute Error (MAE)"
        ].iloc[0]
        self.assertLess(perfect_mae, imperfect_mae)

    def test_zero_values_handling(self):
        # Create dataset with zero values but maintain same features as training
        zero_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [0.5, 1.5, 2.5],
                "target": [0, 2, 4],  # Include zero
            }
        )

        # Initialize ValidMind objects
        vm_zero_dataset = vm.init_dataset(
            input_id="zero_dataset",
            dataset=zero_df,
            target_column="target",
            __log=False,
        )

        # Use existing model
        vm_zero_dataset.assign_predictions(self.vm_model1)

        # Run comparison
        result = RegressionErrorsComparison([vm_zero_dataset], [self.vm_model1])

        # Check that MAPE is None due to zero values
        self.assertIsNone(result["Mean Absolute Percentage Error (MAPE)"].iloc[0])
