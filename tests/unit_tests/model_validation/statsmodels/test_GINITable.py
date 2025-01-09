import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.statsmodels.GINITable import GINITable


class TestGINITable(unittest.TestCase):
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

    def test_returns_dataframe_and_rawdata(self):
        # Run the function
        result, raw_data = GINITable(self.vm_dataset, self.vm_model)

        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if raw_data is RawData instance
        self.assertIsInstance(raw_data, RawData)

        # Check if DataFrame has expected columns
        expected_columns = ["AUC", "GINI", "KS"]
        self.assertTrue(all(col in result.columns for col in expected_columns))

        # Check if DataFrame has exactly one row
        self.assertEqual(len(result), 1)

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

        # Calculate metrics
        result, _ = GINITable(vm_perfect_dataset, vm_perfect_model)

        # For perfect separation:
        # - AUC should be 1.0
        # - GINI should be 1.0 (2*AUC - 1)
        # - KS should be 1.0
        self.assertAlmostEqual(result["AUC"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(result["GINI"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(result["KS"].iloc[0], 1.0, places=5)

    def test_random_prediction(self):
        # Create a dataset with random predictions
        np.random.seed(42)
        n_samples = 100
        random_df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
                "target": np.random.randint(0, 2, n_samples),
            }
        )

        # Create a model that predicts randomly
        random_model = LogisticRegression()
        random_model.fit(random_df[["feature1", "feature2"]], random_df["target"])

        # Initialize ValidMind objects
        vm_random_dataset = vm.init_dataset(
            input_id="random_dataset",
            dataset=random_df,
            target_column="target",
            __log=False,
        )

        vm_random_model = vm.init_model(
            input_id="random_model",
            model=random_model,
            __log=False,
        )

        # Assign predictions to the dataset
        vm_random_dataset.assign_predictions(vm_random_model)

        # Calculate metrics
        result, _ = GINITable(vm_random_dataset, vm_random_model)

        # For random predictions:
        # - AUC should be close to 0.5
        # - GINI should be close to 0.0 (2*0.5 - 1)
        # - KS should be close to 0.0
        self.assertGreater(result["AUC"].iloc[0], 0.4)
        self.assertLess(result["AUC"].iloc[0], 0.6)
        self.assertGreater(result["GINI"].iloc[0], -0.2)
        self.assertLess(result["GINI"].iloc[0], 0.2)
        self.assertLess(result["KS"].iloc[0], 0.2)

    def test_metric_ranges(self):
        # Test regular case
        result, _ = GINITable(self.vm_dataset, self.vm_model)

        # Check metric ranges
        # AUC should be between 0 and 1
        self.assertGreaterEqual(result["AUC"].iloc[0], 0.0)
        self.assertLessEqual(result["AUC"].iloc[0], 1.0)

        # GINI should be between -1 and 1
        self.assertGreaterEqual(result["GINI"].iloc[0], -1.0)
        self.assertLessEqual(result["GINI"].iloc[0], 1.0)

        # KS should be between 0 and 1
        self.assertGreaterEqual(result["KS"].iloc[0], 0.0)
        self.assertLessEqual(result["KS"].iloc[0], 1.0)

        # GINI should be 2*AUC - 1
        self.assertAlmostEqual(
            result["GINI"].iloc[0], 2 * result["AUC"].iloc[0] - 1, places=5
        )
