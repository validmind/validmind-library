import unittest
import numpy as np
import pandas as pd
import validmind as vm
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from validmind.tests.model_validation.sklearn.ROCCurve import ROCCurve

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc,assignment]


class TestROCCurveMulticlassMissingClass(unittest.TestCase):
    """A training class absent from the evaluated slice now computes (sklearn).

    Uses LogisticRegression so it runs without the xgboost extra. The model is fit
    on four classes; the dataset omits one, which previously tripped the shape
    guard and skipped. Alignment on estimator.classes_ makes it compute instead.
    """

    def setUp(self):
        X, y = make_classification(
            n_samples=400,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        model = LogisticRegression(max_iter=1000).fit(X, y)

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y
        df = df[df["target"] != 3].reset_index(drop=True)

        self.ds = vm.init_dataset(
            input_id="mc_roc_missing", dataset=df, target_column="target", __log=False
        )
        self.model = vm.init_model(
            input_id="mc_roc_missing_model", model=model, __log=False
        )
        self.ds.assign_predictions(self.model)

    def test_missing_class_produces_present_curves_plus_micro(self):
        fig, raw = ROCCurve(self.model, self.ds)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw, vm.RawData)

        names = [t.name for t in fig.data]
        # 3 present-class curves + micro-average + random baseline = 5 traces.
        self.assertEqual(len(fig.data), 5)
        self.assertEqual(sum(n.startswith("Class ") for n in names), 3)
        self.assertTrue(any(n.startswith("Micro-average") for n in names))

        # Per-class RawData keyed by the present classes (no absent class 3) + micro.
        self.assertEqual(set(raw.auc), {"0", "1", "2", "micro"})


@unittest.skipUnless(XGBClassifier is not None, "xgboost optional extra required")
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


@unittest.skipUnless(XGBClassifier is not None, "xgboost optional extra required")
class TestROCCurveMulticlass(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n_samples = 900
        X = np.random.randn(n_samples, 3)
        # 3-class target with real signal so per-class AUCs beat random.
        score = np.stack([X[:, 0], X[:, 1], X[:, 2]], axis=1)
        y = (score + np.random.randn(n_samples, 3) * 0.3).argmax(axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )

        test_df = pd.DataFrame(
            {
                "f1": X_test[:, 0],
                "f2": X_test[:, 1],
                "f3": X_test[:, 2],
                "target": y_test,
            }
        )

        self.vm_test_ds = vm.init_dataset(
            input_id="mc_test", dataset=test_df, target_column="target", __log=False
        )
        model = XGBClassifier()
        model.fit(X_train, y_train)
        self.vm_model = vm.init_model(input_id="mc_model", model=model, __log=False)
        self.vm_test_ds.assign_predictions(self.vm_model)

    def test_multiclass_one_vs_rest_traces(self):
        fig, raw = ROCCurve(self.vm_model, self.vm_test_ds)

        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw, vm.RawData)

        names = [t.name for t in fig.data]
        # 3 per-class curves + micro-average + random baseline = 5 traces.
        self.assertEqual(len(fig.data), 5)
        self.assertEqual(sum(n.startswith("Class ") for n in names), 3)
        self.assertTrue(any(n.startswith("Micro-average") for n in names))
        self.assertTrue(any(n == "Random (AUC = 0.5)" for n in names))

        # Per-class RawData keyed by class label + micro; AUCs beat random.
        self.assertEqual(set(raw.auc) - {"micro"}, {"0", "1", "2"})
        for key in ("0", "1", "2", "micro"):
            self.assertGreater(raw.auc[key], 0.5)
