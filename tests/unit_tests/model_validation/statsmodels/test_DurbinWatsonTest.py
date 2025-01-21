import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import validmind as vm
from validmind.tests.model_validation.statsmodels.DurbinWatsonTest import (
    DurbinWatsonTest,
)


class TestDurbinWatsonTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample time series dataset
        np.random.seed(42)
        n_samples = 100

        # Create time-dependent feature
        t = np.linspace(0, 10, n_samples)

        # Create features with some autocorrelation
        X = np.column_stack(
            [
                np.sin(t) + np.random.normal(0, 0.1, n_samples),
                np.cos(t) + np.random.normal(0, 0.1, n_samples),
            ]
        )

        # Create target with some autocorrelation
        y = (
            2 * X[:, 0]
            + 0.5 * X[:, 1]
            + 0.3 * np.roll(X[:, 0], 1)
            + np.random.normal(0, 0.1, n_samples)
        )

        # Create DataFrame
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

    def test_returns_dataframe_and_rawdata(self):
        # Run the function
        results = DurbinWatsonTest(self.vm_dataset, self.vm_model)

        # Check if results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = ["dw_statistic", "threshold", "autocorrelation"]
        self.assertTrue(all(col in results.columns for col in expected_columns))

        # Check if DataFrame has exactly one row
        self.assertEqual(len(results), 1)

    def test_no_autocorrelation(self):
        # Create a dataset with no autocorrelation
        n_samples = 100
        X = np.random.normal(0, 1, (n_samples, 2))
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)

        no_auto_df = pd.DataFrame(
            {"feature1": X[:, 0], "feature2": X[:, 1], "target": y}
        )

        # Create and train model
        model = LinearRegression()
        model.fit(no_auto_df[["feature1", "feature2"]], no_auto_df["target"])

        # Initialize ValidMind objects
        vm_no_auto_dataset = vm.init_dataset(
            input_id="no_auto_dataset",
            dataset=no_auto_df,
            target_column="target",
            __log=False,
        )

        vm_no_auto_model = vm.init_model(
            input_id="no_auto_model",
            model=model,
            __log=False,
        )

        # Assign predictions
        vm_no_auto_dataset.assign_predictions(vm_no_auto_model)

        # Run the function
        results = DurbinWatsonTest(vm_no_auto_dataset, vm_no_auto_model)

        # Check if results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)

        # Check if DataFrame has expected columns
        expected_columns = ["dw_statistic", "threshold", "autocorrelation"]
        self.assertTrue(all(col in results.columns for col in expected_columns))

        # Check if DataFrame has exactly one row
        self.assertEqual(len(results), 1)

        # For no autocorrelation:
        # - DW statistic should be close to 2 (typically between 1.5 and 2.5)
        # - Should be labeled as "No autocorrelation"
        dw_stat = results["dw_statistic"].iloc[0]
        self.assertGreater(
            dw_stat, 1.5, f"DW statistic {dw_stat} indicates positive autocorrelation"
        )
        self.assertLess(
            dw_stat, 2.5, f"DW statistic {dw_stat} indicates negative autocorrelation"
        )
        self.assertEqual(results["autocorrelation"].iloc[0], "No autocorrelation")
        self.assertEqual(results["threshold"].iloc[0], "[1.5, 2.5]")
