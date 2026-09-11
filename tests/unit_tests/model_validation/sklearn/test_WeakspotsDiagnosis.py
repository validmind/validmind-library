import functools
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

    def _all_positive_pair(self, name):
        # Deterministic dataset for pass/fail scoping tests. Every feature slice
        # is 2/3 positive and every row is predicted positive, so for each slice:
        #   Accuracy = Precision = 0.667, Recall = 1.0, F1 = 0.8.
        # With the default thresholds this fails only on Accuracy (< 0.75), while
        # F1 (0.8) clears its 0.7 threshold. That makes an all-four pass/fail
        # (False) disagree with an F1-only pass/fail (True), so the test tells
        # correct scoping apart from mere crash avoidance.
        n = 120
        pattern = ([1, 1, 0] * (n // 3))[:n]
        preds = [1] * n
        train, model = _dataset_with_predictions(f"{name}_train", pattern, preds)
        test, _ = _dataset_with_predictions(f"{name}_test", pattern, preds, model=model)
        return train, test, model

    def test_metrics_subset_without_thresholds_runs(self):
        # Regression for sc-17267: a custom metrics dict that is a strict subset of
        # the default metric names with no thresholds= previously raised
        # "Unable to coerce to Series, length must be 1: given 4" because
        # pass_thresholds still carried all four default keys.
        train, test, model = self._all_positive_pair("subset_run")
        result = WeakspotsDiagnosis(
            datasets=[train, test],
            model=model,
            features_columns=["f1"],
            metrics={"f1": skm.f1_score},
        )
        self.assertGreater(len(result[0]), 0)
        self.assertIsInstance(result[-1], bool)

    def test_metrics_subset_scopes_pass_fail_to_that_metric(self):
        # The F1-only pass/fail must reflect only the F1 threshold. On this data
        # the all-four default fails (Accuracy 0.667 < 0.75) while F1 (0.8) passes,
        # so the two runs must disagree.
        train, test, model = self._all_positive_pair("subset_scope")

        default = WeakspotsDiagnosis(
            datasets=[train, test], model=model, features_columns=["f1"]
        )
        f1_only = WeakspotsDiagnosis(
            datasets=[train, test],
            model=model,
            features_columns=["f1"],
            metrics={"f1": skm.f1_score},
        )

        self.assertFalse(default[-1])
        self.assertTrue(f1_only[-1])

    def test_metrics_subset_with_full_thresholds_runs(self):
        # Supplying thresholds for metrics that are not computed must not raise:
        # pass_columns narrows to the intersection with the requested metrics.
        train, test, model = self._all_positive_pair("subset_full_thresh")
        result = WeakspotsDiagnosis(
            datasets=[train, test],
            model=model,
            features_columns=["f1"],
            metrics={"f1": skm.f1_score},
            thresholds={"accuracy": 0.9, "f1": 0.7},
        )
        # Only F1 (0.8 >= 0.7) is in scope, so the untriggered Accuracy threshold
        # (0.9) must not fail the test.
        self.assertTrue(result[-1])

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


class TestWeakspotsDiagnosisCustomMetrics(unittest.TestCase):
    """Follow-up to the #534 review: user-supplied custom metrics must keep their
    own averaging. ``functools.partial(f1_score, average="weighted")`` still exposes
    ``average`` via ``inspect.signature``, so the old unconditional ``_apply_averaging``
    silently rebound it to the resolved default (macro/binary)."""

    @staticmethod
    def _first_nonempty_slice(dataset, model, feature):
        """Return (region, y_true, y_pred) for the first non-empty bin of a feature,
        mirroring WeakspotsDiagnosis' own binning so recomputation lines up."""
        target = dataset.target_column
        pred_col = dataset.prediction_column(model)
        binned = dataset._df.copy()
        binned["bin"] = pd.cut(binned[feature], bins=10)
        for region, slice_df in binned.groupby("bin", observed=True):
            if slice_df.empty:
                continue
            y_true = slice_df[target].values
            y_pred = slice_df[pred_col].astype(slice_df[target].dtype).values
            return region, y_true, y_pred
        return None, None, None

    def test_custom_metric_keeps_user_averaging(self):
        # Multiclass target: the resolved default is macro averaging, so the old code
        # rebound the user's weighted partial to macro. The results table must carry
        # the user's *weighted* F1, not macro.
        train, test, model = _train_test_pair("custom_weighted", [0, 1, 2])
        weighted_f1 = functools.partial(skm.f1_score, average="weighted")
        result = WeakspotsDiagnosis(
            datasets=[train, test],
            model=model,
            metrics={"f1": weighted_f1},
            thresholds={"f1": 0.5},
        )
        df = result[0]

        feature = train.feature_columns[0]
        region, y_true, y_pred = self._first_nonempty_slice(train, model, feature)
        self.assertIsNotNone(region)
        expected_weighted = weighted_f1(y_true, y_pred)
        expected_macro = skm.f1_score(y_true, y_pred, average="macro")
        # The scenario is only meaningful when the two averagings disagree.
        self.assertNotAlmostEqual(expected_weighted, expected_macro)

        row = df[
            (df["Feature"] == feature)
            & (df["Slice"] == str(region))
            & (df["Dataset"] == train.input_id)
        ]
        self.assertAlmostEqual(row["F1"].iloc[0], expected_weighted)

    def test_default_metrics_still_get_macro_rebinding(self):
        # The defaults path must keep resolving/binding averaging: on a multiclass
        # target the default F1 must equal macro (sklearn's binary default would
        # raise), confirming the fix only skips rebinding for custom metrics.
        train, test, model = _train_test_pair("default_macro", [0, 1, 2])
        result = WeakspotsDiagnosis(datasets=[train, test], model=model)
        df = result[0]

        feature = train.feature_columns[0]
        region, y_true, y_pred = self._first_nonempty_slice(train, model, feature)
        self.assertIsNotNone(region)
        expected_macro = skm.f1_score(y_true, y_pred, average="macro", zero_division=0)

        row = df[
            (df["Feature"] == feature)
            & (df["Slice"] == str(region))
            & (df["Dataset"] == train.input_id)
        ]
        self.assertAlmostEqual(row["F1"].iloc[0], expected_macro)


if __name__ == "__main__":
    unittest.main()
