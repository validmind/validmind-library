import unittest

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

import validmind as vm
from validmind.tests.model_validation.sklearn._diagnosis_metrics import (
    apply_averaging,
    bind_averaging,
    full_labels,
    multiclass_auc,
    resolve_averaging,
)


def _dataset_with_predictions(input_id, y_true, y_pred, model=None):
    """Build a VMDataset with injected predictions to control label sets exactly."""
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


class TestFullLabelsAndResolveAveraging(unittest.TestCase):
    def test_full_labels_unions_true_and_predicted(self):
        # class 4 appears only in predictions; it must still be part of the space.
        ds, model = _dataset_with_predictions(
            "fl", [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 4, 2]
        )
        self.assertEqual(full_labels([ds], model), [0, 1, 2, 4])

    def test_resolve_binary_non_standard(self):
        ds, model = _dataset_with_predictions("rb", [0, 4, 0, 4], [0, 4, 4, 0])
        self.assertEqual(resolve_averaging([ds], model), ("binary", 4))

    def test_resolve_standard_binary(self):
        ds, model = _dataset_with_predictions("rs", [0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(resolve_averaging([ds], model), ("binary", 1))

    def test_resolve_multiclass(self):
        ds, model = _dataset_with_predictions(
            "rm", [0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 1, 2]
        )
        self.assertEqual(resolve_averaging([ds], model), ("macro", None))


class TestBindAveraging(unittest.TestCase):
    def test_none_average_is_passthrough(self):
        self.assertIs(bind_averaging(skm.f1_score, None, None), skm.f1_score)

    def test_accuracy_passthrough(self):
        # accuracy has no `average` parameter, so it must be returned unchanged.
        self.assertIs(
            bind_averaging(skm.accuracy_score, "macro", None), skm.accuracy_score
        )

    def test_binary_binds_pos_label(self):
        fn = bind_averaging(skm.precision_score, "binary", 4)
        y_true = np.array([0, 4, 0, 4])
        y_pred = np.array([0, 4, 4, 4])
        self.assertEqual(
            fn(y_true, y_pred),
            skm.precision_score(y_true, y_pred, average="binary", pos_label=4),
        )

    def test_macro_omits_pos_label(self):
        fn = bind_averaging(skm.f1_score, "macro", None)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        self.assertEqual(
            fn(y_true, y_pred), skm.f1_score(y_true, y_pred, average="macro")
        )

    def test_apply_averaging_maps_over_dict(self):
        prepared = apply_averaging(
            {"acc": skm.accuracy_score, "f1": skm.f1_score}, "macro", None
        )
        self.assertIs(prepared["acc"], skm.accuracy_score)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        self.assertEqual(
            prepared["f1"](y_true, y_pred),
            skm.f1_score(y_true, y_pred, average="macro"),
        )


class TestMulticlassAuc(unittest.TestCase):
    LABELS = [0, 1, 2, 3, 4]

    def test_matches_label_binarize_when_all_classes_present(self):
        # When every class appears, the helper must equal the label-binarize AUC
        # convention used by ClassifierPerformance.
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 2, 4, 0, 3, 2, 3, 4])
        lb = LabelBinarizer().fit(y_true)
        expected = skm.roc_auc_score(
            lb.transform(y_true), lb.transform(y_pred), average="macro"
        )
        self.assertAlmostEqual(multiclass_auc(y_true, y_pred, self.LABELS), expected)

    def test_subset_slice_does_not_raise(self):
        # A slice with only two of the classes (0 and 4) plus a stray prediction.
        y_true = np.array([0, 0, 4, 4, 0])
        y_pred = np.array([0, 2, 4, 0, 4])
        value = multiclass_auc(y_true, y_pred, self.LABELS)
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_single_class_slice_returns_zero(self):
        # No class has both positive and negative examples -> nothing scorable.
        y_true = np.array([4, 4, 4])
        y_pred = np.array([4, 0, 4])
        self.assertEqual(multiclass_auc(y_true, y_pred, self.LABELS), 0.0)


if __name__ == "__main__":
    unittest.main()
