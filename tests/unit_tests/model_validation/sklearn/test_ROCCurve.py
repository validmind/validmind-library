import unittest
import numpy as np
import pandas as pd
import validmind as vm
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from validmind.tests.model_validation.sklearn.ROCCurve import ROCCurve


class TestROCCurve(unittest.TestCase):
    def setUp(self):
        # Create binary classification test dataset
        np.random.seed(42)
        n_samples = 1000

        # Create features that have some predictive power
        X = np.random.randn(n_samples, 2)
        # Create target with actual relationship to features
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        # First split into train_val and test (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Then split train_val into train and validation (75/25 of 80 = 60/20 of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )

        # Create full dataset with features and target
        data = {"feature1": X_train[:, 0], "feature2": X_train[:, 1], "target": y_train}
        train_df = pd.DataFrame(data)

        # Create test dataset
        data = {"feature1": X_test[:, 0], "feature2": X_test[:, 1], "target": y_test}
        test_df = pd.DataFrame(data)

        self.vm_train_ds = vm.init_dataset(
            input_id="train_dataset",
            dataset=train_df,
            target_column="target",
            __log=False,
        )

        self.vm_test_ds = vm.init_dataset(
            input_id="test_dataset",
            dataset=test_df,
            target_column="target",
            __log=False,
        )

        # Create and train XGBoost model
        xgb_model = XGBClassifier(early_stopping_rounds=10)
        xgb_model.set_params(eval_metric=["error", "logloss", "auc"])
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        self.vm_model = vm.init_model(
            input_id="xgb_model", model=xgb_model, __log=False
        )

        # Assign predictions to the datasets
        self.vm_train_ds.assign_predictions(self.vm_model)
        self.vm_test_ds.assign_predictions(self.vm_model)

    def test_roc_curve_structure(self):
        result = ROCCurve(self.vm_model, self.vm_test_ds)

        # Check return type is tuple with RawData and Figure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], go.Figure)
        self.assertIsInstance(result[1], vm.RawData)

        # Get the figure from the tuple
        fig = result[0]

        # Check figure has two traces (ROC curve and random baseline)
        self.assertEqual(len(fig.data), 2)

        # Check trace types and names
        self.assertEqual(fig.data[0].mode, "lines")
        self.assertIn("ROC curve (AUC = ", fig.data[0].name)
        self.assertEqual(fig.data[1].name, "Random (AUC = 0.5)")

        # Check AUC score is better than random
        auc = float(fig.data[0].name.split("=")[1].strip().rstrip(")"))
        self.assertGreater(auc, 0.5)

    def test_perfect_separation(self):
        # Create perfectly separable dataset
        X = np.random.randn(1000, 2)
        y = (X[:, 0] > 0).astype(int)  # Perfect separation based on first feature

        # Split into train_val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Split train_val into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )

        # Create train dataset
        data = {"feature1": X_train[:, 0], "feature2": X_train[:, 1], "target": y_train}
        train_df = pd.DataFrame(data)

        # Create test dataset
        data = {"feature1": X_test[:, 0], "feature2": X_test[:, 1], "target": y_test}
        test_df = pd.DataFrame(data)

        vm_train_ds = vm.init_dataset(
            input_id="train_dataset",
            dataset=train_df,
            target_column="target",
            __log=False,
        )

        vm_test_ds = vm.init_dataset(
            input_id="test_dataset",
            dataset=test_df,
            target_column="target",
            __log=False,
        )

        # Train model on perfectly separable data
        perfect_model = XGBClassifier(early_stopping_rounds=10)
        perfect_model.set_params(eval_metric=["error", "logloss", "auc"])
        perfect_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        vm_perfect_model = vm.init_model(
            input_id="perfect_model", model=perfect_model, __log=False
        )

        # Assign predictions to the datasets
        vm_train_ds.assign_predictions(vm_perfect_model)
        vm_test_ds.assign_predictions(vm_perfect_model)

        fig, _ = ROCCurve(vm_perfect_model, vm_test_ds)

        # Check AUC score (should be very close to 1.0)
        auc = float(fig.data[0].name.split("=")[1].strip().rstrip(")"))
        self.assertGreater(auc, 0.95)
