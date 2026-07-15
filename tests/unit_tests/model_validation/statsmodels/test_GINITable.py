import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import validmind as vm
from validmind import RawData
from validmind.errors import SkipTestError
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
        results, raw_data = GINITable(self.vm_dataset, self.vm_model)

        # Check if results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)

        # Check if raw_data is a RawData object
        self.assertIsInstance(raw_data, RawData)

        # Check if DataFrame has expected columns
        expected_columns = ["AUC", "GINI", "KS"]
        self.assertTrue(all(col in results.columns for col in expected_columns))

        # Check if DataFrame has exactly one row
        self.assertEqual(len(results), 1)

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
        results, raw_data = GINITable(vm_perfect_dataset, vm_perfect_model)

        # For perfect separation:
        # - AUC should be 1.0
        # - GINI should be 1.0 (2*AUC - 1)
        # - KS should be 1.0
        self.assertAlmostEqual(results["AUC"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(results["GINI"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(results["KS"].iloc[0], 1.0, places=5)

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
        results, raw_data = GINITable(vm_random_dataset, vm_random_model)

        # For random predictions:
        # - AUC should be close to 0.5
        # - GINI should be close to 0.0 (2*0.5 - 1)
        # - KS should be close to 0.0
        self.assertGreater(results["AUC"].iloc[0], 0.4)
        self.assertLess(results["AUC"].iloc[0], 0.6)
        self.assertGreater(results["GINI"].iloc[0], -0.2)
        self.assertLess(results["GINI"].iloc[0], 0.2)
        self.assertLess(results["KS"].iloc[0], 0.2)

    def test_metric_ranges(self):
        # Test regular case
        results, raw_data = GINITable(self.vm_dataset, self.vm_model)

        # Check metric ranges
        # AUC should be between 0 and 1
        self.assertGreaterEqual(results["AUC"].iloc[0], 0.0)
        self.assertLessEqual(results["AUC"].iloc[0], 1.0)

        # GINI should be between -1 and 1
        self.assertGreaterEqual(results["GINI"].iloc[0], -1.0)
        self.assertLessEqual(results["GINI"].iloc[0], 1.0)

        # KS should be between 0 and 1
        self.assertGreaterEqual(results["KS"].iloc[0], 0.0)
        self.assertLessEqual(results["KS"].iloc[0], 1.0)

        # GINI should be 2*AUC - 1
        self.assertAlmostEqual(
            results["GINI"].iloc[0], 2 * results["AUC"].iloc[0] - 1, places=5
        )


class TestGINITableMulticlass(unittest.TestCase):
    def setUp(self):
        # 3-class target with real signal so per-class AUCs beat random.
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        self.vm_dataset = vm.init_dataset(
            input_id="mc_dataset", dataset=df, target_column="target", __log=False
        )
        self.vm_model = vm.init_model(
            input_id="mc_model", model=model, __log=False
        )
        self.vm_dataset.assign_predictions(self.vm_model)

    def test_multiclass_one_vs_rest_rows(self):
        results, raw_data = GINITable(self.vm_dataset, self.vm_model)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(raw_data, RawData)

        # One row per class + a micro-average row.
        self.assertEqual(set(results["Class"]), {"0", "1", "2", "micro"})
        self.assertEqual(len(results), 4)
        for col in ("Class", "AUC", "GINI", "KS"):
            self.assertIn(col, results.columns)

        # Per-class and micro AUCs beat random, and GINI == 2*AUC - 1.
        for _, row in results.iterrows():
            self.assertGreater(row["AUC"], 0.5)
            self.assertAlmostEqual(row["GINI"], 2 * row["AUC"] - 1, places=5)
            self.assertGreaterEqual(row["KS"], 0.0)
            self.assertLessEqual(row["KS"], 1.0)

        # RawData carries per-class + micro fpr/tpr curves.
        self.assertEqual(set(raw_data.fpr), {"0", "1", "2", "micro"})
        self.assertEqual(set(raw_data.tpr), {"0", "1", "2", "micro"})

    def test_multiclass_without_predict_proba_is_skipped(self):
        # A model that can't produce a per-class probability matrix must skip,
        # not crash. Shadow predict_proba on the estimator instance (predictions
        # were already assigned in setUp, so nothing downstream needs it).
        self.vm_model.model.predict_proba = None
        with self.assertRaises(SkipTestError):
            GINITable(self.vm_dataset, self.vm_model)
