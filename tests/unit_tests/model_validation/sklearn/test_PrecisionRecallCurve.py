import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

import validmind as vm
from validmind.tests.model_validation.sklearn.PrecisionRecallCurve import (
    PrecisionRecallCurve,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc,assignment]


@unittest.skipUnless(XGBClassifier is not None, "xgboost optional extra required")
class TestPrecisionRecallCurveBinary(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 800
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n) * 0.1 > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        test_df = pd.DataFrame(
            {"f1": X_test[:, 0], "f2": X_test[:, 1], "target": y_test}
        )
        self.ds = vm.init_dataset(
            input_id="bin_test", dataset=test_df, target_column="target", __log=False
        )
        model = XGBClassifier().fit(X_train, y_train)
        self.model = vm.init_model(input_id="bin_model", model=model, __log=False)
        self.ds.assign_predictions(self.model)

    def test_binary_single_curve(self):
        fig, raw = PrecisionRecallCurve(self.model, self.ds)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw, vm.RawData)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].name, "Precision-Recall Curve")


@unittest.skipUnless(XGBClassifier is not None, "xgboost optional extra required")
class TestPrecisionRecallCurveMulticlass(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 900
        X = np.random.randn(n, 3)
        y = (
            np.stack([X[:, 0], X[:, 1], X[:, 2]], axis=1) + np.random.randn(n, 3) * 0.3
        ).argmax(axis=1)
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
        self.ds = vm.init_dataset(
            input_id="mc_pr_test", dataset=test_df, target_column="target", __log=False
        )
        model = XGBClassifier().fit(X_train, y_train)
        self.model = vm.init_model(input_id="mc_pr_model", model=model, __log=False)
        self.ds.assign_predictions(self.model)

    def test_multiclass_one_vs_rest_traces(self):
        fig, raw = PrecisionRecallCurve(self.model, self.ds)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw, vm.RawData)

        names = [t.name for t in fig.data]
        # 3 per-class curves + micro-average = 4 traces (no random baseline for PR).
        self.assertEqual(len(fig.data), 4)
        self.assertEqual(sum(n.startswith("Class ") for n in names), 3)
        self.assertTrue(any(n.startswith("Micro-average") for n in names))

        self.assertEqual(set(raw.average_precision) - {"micro"}, {"0", "1", "2"})
        for key in ("0", "1", "2", "micro"):
            self.assertGreater(raw.average_precision[key], 0.5)
