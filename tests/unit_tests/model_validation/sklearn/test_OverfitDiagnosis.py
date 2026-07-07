import functools
import unittest

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import validmind as vm
from validmind.tests.model_validation.sklearn.OverfitDiagnosis import (
    OverfitDiagnosis,
    _classification_metric_fn,
)


def _classification_datasets(input_id, labels, seed=0, n=160):
    """Build train/test VMDatasets with a fitted classifier.

    Predictions and probabilities are computed by the model (not injected) so
    ``OverfitDiagnosis`` detects a classification task via the probability
    column. One feature is correlated with the class so feature bins carry a
    subset of the classes.
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

    train_ds.assign_predictions(model=vm_model)
    test_ds.assign_predictions(model=vm_model)
    return train_ds, test_ds, vm_model


class TestOverfitClassificationMetricFn(unittest.TestCase):
    """Unit tests for the averaging strategy selected from the global labels."""

    def test_multiclass_labels_use_macro(self):
        labels = np.array([0, 2, 4])
        fn = _classification_metric_fn("f1", labels)

        self.assertIsInstance(fn, functools.partial)
        self.assertIs(fn.func, metrics.f1_score)
        self.assertEqual(fn.keywords["average"], "macro")
        self.assertEqual(fn.keywords["zero_division"], 0)
        self.assertTrue(np.array_equal(fn.keywords["labels"], labels))

    def test_non_standard_binary_uses_pos_label(self):
        fn = _classification_metric_fn("precision", np.array([0, 4]))

        self.assertIsInstance(fn, functools.partial)
        self.assertIs(fn.func, metrics.precision_score)
        self.assertEqual(fn.keywords, {"pos_label": 4})

    def test_conventional_binary_left_unchanged(self):
        self.assertIs(
            _classification_metric_fn("recall", np.array([0, 1])), metrics.recall_score
        )

    def test_non_prf_metrics_left_unchanged(self):
        # auc/accuracy never take averaging kwargs and must not be wrapped.
        self.assertIs(
            _classification_metric_fn("auc", np.array([0, 2, 4])),
            metrics.roc_auc_score,
        )
        self.assertIs(
            _classification_metric_fn("accuracy", np.array([0, 2, 4])),
            metrics.accuracy_score,
        )


class TestOverfitDiagnosisMulticlass(unittest.TestCase):
    """Regression tests for ZD-704 sibling exposure (explicit f1 selection)."""

    def test_multiclass_f1(self):
        train_ds, test_ds, model = _classification_datasets(
            "ovf_multi", [0, 2, 4], seed=1
        )
        result = OverfitDiagnosis(
            model=model, datasets=[train_ds, test_ds], metric="f1"
        )
        self.assertIn("Overfit Diagnosis", result[0])

    def test_binary_f1_without_one(self):
        train_ds, test_ds, model = _classification_datasets(
            "ovf_binary04", [0, 4], seed=2
        )
        result = OverfitDiagnosis(
            model=model, datasets=[train_ds, test_ds], metric="f1"
        )
        self.assertIn("Overfit Diagnosis", result[0])


if __name__ == "__main__":
    unittest.main()
