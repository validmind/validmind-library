import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import validmind as vm
from validmind.tests.model_validation.sklearn.RegressionErrors import RegressionErrors


class TestRegressionErrors(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample dataset with known relationships
        np.random.seed(42)
        n_samples = 100

        # Create features
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

    def test_returns_dataframe_and_raw_data(self):
        # Run the function
        results = RegressionErrors(self.vm_model, self.vm_dataset)

        # Check if results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = [
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Percentage Error (MAPE)",
            "Mean Bias Deviation (MBD)",
        ]
        self.assertTrue(all(col in results.columns for col in expected_columns))

        # Check if DataFrame has exactly one row
        self.assertEqual(len(results), 1)

    def test_error_metrics_range(self):
        results = RegressionErrors(self.vm_model, self.vm_dataset)

        # All error metrics should be non-negative (except MBD)
        self.assertGreaterEqual(results["Mean Absolute Error (MAE)"].iloc[0], 0)
        self.assertGreaterEqual(results["Mean Squared Error (MSE)"].iloc[0], 0)
        self.assertGreaterEqual(results["Root Mean Squared Error (RMSE)"].iloc[0], 0)
        self.assertGreaterEqual(
            results["Mean Absolute Percentage Error (MAPE)"].iloc[0], 0
        )

        # Check if RMSE is square root of MSE
        mse = results["Mean Squared Error (MSE)"].iloc[0]
        rmse = results["Root Mean Squared Error (RMSE)"].iloc[0]
        self.assertAlmostEqual(rmse, np.sqrt(mse), places=5)

    def test_perfect_prediction(self):
        # Create a perfect prediction scenario
        perfect_df = pd.DataFrame(
            {"feature1": [1, 2, 3], "target": [2, 4, 6]}  # perfect linear relationship
        )

        # Create and train a model that will give perfect predictions
        perfect_model = LinearRegression()
        perfect_model.fit(perfect_df[["feature1"]], perfect_df["target"])

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

        # Assign predictions to the perfect dataset
        vm_perfect_dataset.assign_predictions(vm_perfect_model)

        # Calculate errors
        results = RegressionErrors(vm_perfect_model, vm_perfect_dataset)

        # All error metrics should be very close to 0
        self.assertAlmostEqual(
            results["Mean Absolute Error (MAE)"].iloc[0], 0, places=5
        )
        self.assertAlmostEqual(results["Mean Squared Error (MSE)"].iloc[0], 0, places=5)
        self.assertAlmostEqual(
            results["Root Mean Squared Error (RMSE)"].iloc[0], 0, places=5
        )
        self.assertAlmostEqual(
            results["Mean Bias Deviation (MBD)"].iloc[0], 0, places=5
        )

    def test_error_metrics_consistency(self):
        results = RegressionErrors(self.vm_model, self.vm_dataset)

        # MSE should be greater than or equal to MAE squared
        mae = results["Mean Absolute Error (MAE)"].iloc[0]
        mse = results["Mean Squared Error (MSE)"].iloc[0]
        self.assertGreaterEqual(mse, mae * mae)

        # RMSE should be greater than or equal to MAE
        rmse = results["Root Mean Squared Error (RMSE)"].iloc[0]
        self.assertGreaterEqual(rmse, mae)
