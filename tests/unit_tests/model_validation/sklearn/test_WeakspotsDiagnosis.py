import unittest

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression

import validmind as vm
from validmind.tests.model_validation.sklearn.WeakspotsDiagnosis import (
    WeakspotsDiagnosis,
    _apply_averaging,
    _prepare_metrics_and_thresholds,
    _resolve_averaging,
)


def _dataset_with_predictions(input_id, y_true, y_pred, model=None):
    """Build a VMDataset with two numeric features and injected predictions.

    Predictions are supplied via ``prediction_values`` so the model's ``predict``
    is never called and the true/predicted label sets can be controlled exactly
    (including binary targets encoded outside ``{0, 1}``). Passing an existing
    ``model`` lets a train/test pair share one VMModel input.
    """
    n = len(y_true)
    rng = np.random.RandomState(abs(hash(input_id)) % (2**32))
    df = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, n),
            "f2": rng.randn(n),
            "target": y_true,
        }
    )
    dataset = vm.init_dataset(
        input_id=input_id, dataset=df, target_column="target", __log=False
    )
    if model is None:
        estimator = LogisticRegression(max_iter=1000)
        estimator.fit(df[["f1", "f2"]].to_numpy(), np.array(y_true))
        model = vm.init_model(
            input_id=f"{input_id}_model", model=estimator, __log=False
        )
    dataset.assign_predictions(model=model, prediction_values=y_pred)
    return dataset, model


def _train_test_pair(name, labels, seed=7, n=48):
    rng = np.random.RandomState(seed)
    train, model = _dataset_with_predictions(
        f"{name}_train",
        rng.choice(labels, size=n).tolist(),
        rng.choice(labels, size=n).tolist(),
    )
    test, _ = _dataset_with_predictions(
        f"{name}_test",
        rng.choice(labels, size=n).tolist(),
        rng.choice(labels, size=n).tolist(),
        model=model,
    )
    return train, test, model


class TestWeakspotsDiagnosisThresholds(unittest.TestCase):
    def test_partial_thresholds_use_defaults_for_plotting(self):
        _, plot_thresholds, pass_thresholds = _prepare_metrics_and_thresholds(
            metrics=None,
            thresholds={"accuracy": 0.65},
        )

        self.assertEqual(pass_thresholds, {"Accuracy": 0.65})
        self.assertEqual(plot_thresholds["Accuracy"], 0.65)
        self.assertEqual(plot_thresholds["Precision"], 0.5)
        self.assertEqual(plot_thresholds["Recall"], 0.5)
        self.assertEqual(plot_thresholds["F1"], 0.7)

    def test_partial_thresholds_subset_for_pass_fail(self):
        _, _, pass_thresholds = _prepare_metrics_and_thresholds(
            metrics=None,
            thresholds={"accuracy": 0.75, "f1": 0.55},
        )

        self.assertEqual(set(pass_thresholds.keys()), {"Accuracy", "F1"})


class TestResolveAveraging(unittest.TestCase):
    def test_binary_non_standard_encoding_selects_positive_label(self):
        # ZD-704 follow-up: binary target encoded as {0, 4}. sklearn's default
        # pos_label=1 is not a valid label, so the positive label must be resolved
        # to the largest class value (4).
        train, test, model = _train_test_pair("resolve_04", [0, 4])
        self.assertEqual(_resolve_averaging([train, test], model), ("binary", 4))

    def test_standard_binary_keeps_pos_label_one(self):
        # {0, 1} must resolve to sklearn's own default so existing results are
        # unchanged.
        train, test, model = _train_test_pair("resolve_01", [0, 1])
        self.assertEqual(_resolve_averaging([train, test], model), ("binary", 1))

    def test_multiclass_selects_macro(self):
        train, test, model = _train_test_pair("resolve_mc", [0, 1, 2])
        self.assertEqual(_resolve_averaging([train, test], model), ("macro", None))

    def test_multiclass_detected_from_predictions(self):
        # True labels are binary but the model predicts a third class; the union of
        # true and predicted labels must still be treated as multiclass.
        train, model = _dataset_with_predictions(
            "resolve_pred_mc",
            [0, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 2, 0, 1, 0, 2, 0],  # class 2 never appears in y_true
        )
        self.assertEqual(_resolve_averaging([train], model), ("macro", None))


class TestApplyAveraging(unittest.TestCase):
    def test_binary_binds_average_and_pos_label(self):
        metrics, _, _ = _prepare_metrics_and_thresholds(metrics=None, thresholds=None)
        prepared = _apply_averaging(metrics, "binary", 4)

        y_true = np.array([0, 4, 0, 4])
        y_pred = np.array([0, 4, 4, 4])
        # accuracy has no `average`/`pos_label`, so it is left untouched.
        self.assertIs(prepared["Accuracy"], skm.accuracy_score)
        self.assertEqual(
            prepared["Precision"](y_true, y_pred),
            skm.precision_score(y_true, y_pred, average="binary", pos_label=4),
        )

    def test_macro_does_not_bind_pos_label(self):
        metrics, _, _ = _prepare_metrics_and_thresholds(metrics=None, thresholds=None)
        prepared = _apply_averaging(metrics, "macro", None)

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        self.assertEqual(
            prepared["F1"](y_true, y_pred),
            skm.f1_score(y_true, y_pred, average="macro"),
        )


class TestWeakspotsDiagnosisEndToEnd(unittest.TestCase):
    def test_binary_non_standard_encoding_runs(self):
        # Regression test for ZD-704: WeakspotsDiagnosis previously raised
        # "pos_label=1 is not a valid label. It should be one of [0, 4]".
        train, test, model = _train_test_pair("e2e_04", [0, 4])
        result = WeakspotsDiagnosis(datasets=[train, test], model=model)
        self.assertGreater(len(result[0]), 0)
        self.assertIsInstance(result[-1], bool)

    def test_multiclass_runs(self):
        # Previously raised "Target is multiclass but average='binary'".
        train, test, model = _train_test_pair("e2e_mc", [0, 1, 2])
        result = WeakspotsDiagnosis(datasets=[train, test], model=model)
        self.assertGreater(len(result[0]), 0)

    def test_multiclass_with_two_class_slice_runs(self):
        # Faithful ZD-704 regression: the target is genuinely multiclass
        # ({0, 1, 2, 3, 4}), but the lowest feature bin contains only two of those
        # classes (0 and 4). The old per-slice scoring saw that slice as "binary"
        # and raised "pos_label=1 is not a valid label. It should be one of [0, 4]".
        # Resolving the averaging mode from the full (multiclass) label set makes
        # every slice use macro averaging, so the sparse slice no longer fails.
        n = 200
        feature = np.linspace(0.0, 10.0, n)
        rng = np.random.RandomState(0)

        def build(input_id, model=None):
            y = rng.choice([0, 1, 2, 3, 4], size=n)
            y[:20] = rng.choice([0, 4], size=20)  # lowest bin has only {0, 4}
            df = pd.DataFrame({"f1": feature, "f2": rng.randn(n), "target": y})
            ds = vm.init_dataset(
                input_id=input_id, dataset=df, target_column="target", __log=False
            )
            if model is None:
                est = LogisticRegression(max_iter=1000)
                est.fit(df[["f1", "f2"]].to_numpy(), y)
                model = vm.init_model(
                    input_id=f"{input_id}_model", model=est, __log=False
                )
            ds.assign_predictions(model=model)
            return ds, model

        train, model = build("e2e_mc_sparse_train")
        test, _ = build("e2e_mc_sparse_test", model=model)

        self.assertEqual(_resolve_averaging([train, test], model), ("macro", None))
        result = WeakspotsDiagnosis(datasets=[train, test], model=model)
        self.assertGreater(len(result[0]), 0)

    def test_standard_binary_matches_default_precision(self):
        # Backward compatibility: for {0, 1} the per-slice Precision values must be
        # identical to scikit-learn's default binary precision so existing reports
        # are unaffected by the fix.
        train, test, model = _train_test_pair("e2e_01", [0, 1])
        result = WeakspotsDiagnosis(datasets=[train, test], model=model)
        df = result[0]

        target = train.target_column
        pred_col = train.prediction_column(model)
        # Recompute the first non-empty slice's precision directly and compare.
        checked = 0
        for feature in train.feature_columns:
            binned = train._df.copy()
            binned["bin"] = pd.cut(binned[feature], bins=10)
            for region, slice_df in binned.groupby("bin", observed=True):
                if slice_df.empty:
                    continue
                y_true = slice_df[target].values
                y_pred = slice_df[pred_col].astype(slice_df[target].dtype).values
                expected = skm.precision_score(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                )
                row = df[
                    (df["Feature"] == feature)
                    & (df["Slice"] == str(region))
                    & (df["Dataset"] == train.input_id)
                ]
                self.assertAlmostEqual(row["Precision"].iloc[0], expected)
                checked += 1
                break
            if checked:
                break
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
