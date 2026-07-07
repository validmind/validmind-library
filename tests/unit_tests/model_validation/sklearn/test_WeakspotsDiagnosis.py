import functools
import unittest

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import validmind as vm
from validmind.tests.model_validation.sklearn.WeakspotsDiagnosis import (
    WeakspotsDiagnosis,
    _averaged_default_metrics,
    _prepare_metrics_and_thresholds,
)


def _train_test_datasets(input_id, labels, seed=0, n=140):
    """Build train/test VMDatasets and a fitted model for a given label set.

    Two numeric features are generated so ``WeakspotsDiagnosis`` has columns to
    bin, with one feature correlated with the class so individual bins carry
    only a subset of the classes (which is what makes per-slice averaging
    crash). Predictions are injected verbatim so the true/predicted label sets
    are controlled exactly and never accidentally collapse to {0, 1}.
    """
    frames = []
    for offset in (0, 100):
        rng = np.random.default_rng(seed + offset)
        y = rng.choice(labels, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "f1": y + rng.normal(0, 0.4, n),
                    "f2": rng.normal(0, 1, n),
                    "target": y,
                }
            )
        )
    train_df, test_df = frames

    train_ds = vm.init_dataset(
        input_id=f"{input_id}_train",
        dataset=train_df,
        target_column="target",
        __log=False,
    )
    test_ds = vm.init_dataset(
        input_id=f"{input_id}_test",
        dataset=test_df,
        target_column="target",
        __log=False,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(train_df[["f1", "f2"]].to_numpy(), train_df["target"].to_numpy())
    vm_model = vm.init_model(input_id=f"{input_id}_model", model=model, __log=False)

    train_ds.assign_predictions(
        model=vm_model,
        prediction_values=model.predict(train_df[["f1", "f2"]].to_numpy()),
    )
    test_ds.assign_predictions(
        model=vm_model,
        prediction_values=model.predict(test_df[["f1", "f2"]].to_numpy()),
    )
    return train_ds, test_ds, vm_model


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


class TestWeakspotsDefaultMetricAveraging(unittest.TestCase):
    """Unit tests for the averaging strategy selected from the global labels.

    Regression tests for ZD-704: sklearn's precision/recall/f1 default to
    ``average="binary", pos_label=1`` which raises for multiclass labels and for
    binary labels that don't contain 1 (e.g. {0, 4}).
    """

    def test_multiclass_labels_use_macro(self):
        labels = np.array([0, 2, 4])
        bound = _averaged_default_metrics(labels)

        for name in ("precision", "recall", "f1"):
            self.assertIsInstance(bound[name], functools.partial)
            self.assertEqual(bound[name].keywords["average"], "macro")
            self.assertEqual(bound[name].keywords["zero_division"], 0)
            self.assertTrue(np.array_equal(bound[name].keywords["labels"], labels))
        # accuracy takes no averaging kwargs and must be left alone
        self.assertIs(bound["accuracy"], metrics.accuracy_score)

    def test_non_standard_binary_uses_pos_label(self):
        # Two classes, neither of which is 1 -- the customer's exact case.
        bound = _averaged_default_metrics(np.array([0, 4]))

        for name in ("precision", "recall", "f1"):
            self.assertIsInstance(bound[name], functools.partial)
            self.assertEqual(bound[name].keywords, {"pos_label": 4})
        self.assertIs(bound["accuracy"], metrics.accuracy_score)

    def test_conventional_binary_left_unchanged(self):
        bound = _averaged_default_metrics(np.array([0, 1]))

        self.assertIs(bound["precision"], metrics.precision_score)
        self.assertIs(bound["recall"], metrics.recall_score)
        self.assertIs(bound["f1"], metrics.f1_score)
        self.assertIs(bound["accuracy"], metrics.accuracy_score)


class TestWeakspotsDiagnosisMulticlass(unittest.TestCase):
    """End-to-end regression tests for ZD-704.

    On unpatched code these raise ``ValueError`` per feature-bin slice; the fix
    decides the averaging once from the global label set so the test runs.
    """

    def test_multiclass_non_contiguous_labels(self):
        train_ds, test_ds, model = _train_test_datasets("wsd_multi", [0, 2, 4], seed=1)
        # Would raise "Target is multiclass but average='binary'" / "pos_label=1
        # is not a valid label" on a per-slice basis before the fix.
        result = WeakspotsDiagnosis(datasets=[train_ds, test_ds], model=model)
        self.assertGreater(len(result), 1)

    def test_binary_labels_without_one(self):
        train_ds, test_ds, model = _train_test_datasets("wsd_binary04", [0, 4], seed=2)
        # Would raise "pos_label=1 is not a valid label. It should be one of
        # [0, 4]" before the fix -- the customer's reported error.
        result = WeakspotsDiagnosis(datasets=[train_ds, test_ds], model=model)
        self.assertGreater(len(result), 1)

    def test_conventional_binary_still_runs(self):
        train_ds, test_ds, model = _train_test_datasets("wsd_binary01", [0, 1], seed=3)
        result = WeakspotsDiagnosis(datasets=[train_ds, test_ds], model=model)
        self.assertGreater(len(result), 1)


if __name__ == "__main__":
    unittest.main()
