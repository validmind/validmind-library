import types
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.model_validation.sklearn._multiclass_proba import (
    MulticlassProba,
    multiclass_proba,
    proba_matrix,
    resolve_class_list,
)


def _fit_model_and_dataset(n_classes=3, drop_class=None, random_state=42):
    """Fit LogisticRegression on ``n_classes`` classes and build a VM dataset.

    ``drop_class`` builds the dataset from a slice with that class removed, while
    the model is still fit on the full class set -- the missing-class scenario.
    """
    X, y = make_classification(
        n_samples=400,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    model = LogisticRegression(max_iter=1000).fit(X, y)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    if drop_class is not None:
        df = df[df["target"] != drop_class].reset_index(drop=True)

    vm_dataset = vm.init_dataset(
        input_id=f"ds_{n_classes}_{drop_class}",
        dataset=df,
        target_column="target",
        __log=False,
    )
    vm_model = vm.init_model(
        input_id=f"model_{n_classes}_{drop_class}", model=model, __log=False
    )
    return vm_model, vm_dataset


class TestResolveClassList(unittest.TestCase):
    def test_uses_estimator_classes_(self):
        # Model trained on 4 classes, dataset slice missing class 3: the column
        # order must follow estimator.classes_ (all four), not np.unique(y).
        model, dataset = _fit_model_and_dataset(n_classes=4, drop_class=3)
        np.testing.assert_array_equal(resolve_class_list(model, dataset), [0, 1, 2, 3])
        # np.unique of the slice would have missed class 3.
        self.assertEqual(sorted(np.unique(dataset.y).tolist()), [0, 1, 2])

    def test_falls_back_to_unique_without_classes_(self):
        # A model object without ``classes_`` falls back to np.unique(dataset.y).
        _, dataset = _fit_model_and_dataset(n_classes=3)
        fake_model = types.SimpleNamespace(model=object())
        np.testing.assert_array_equal(
            resolve_class_list(fake_model, dataset), [0, 1, 2]
        )


class TestMulticlassProbaHappyPath(unittest.TestCase):
    def test_all_classes_present(self):
        model, dataset = _fit_model_and_dataset(n_classes=3)
        result = multiclass_proba(model, dataset, "ROC Curve")

        self.assertIsInstance(result, MulticlassProba)
        np.testing.assert_array_equal(result.class_list, [0, 1, 2])
        self.assertEqual([int(c) for c in result.classes_present], [0, 1, 2])
        self.assertEqual(result.present_indices, [0, 1, 2])

        n = len(dataset.y)
        self.assertEqual(result.y_bin.shape, (n, 3))
        self.assertEqual(result.y_prob.shape, (n, 3))
        # Probability rows sum to 1 (full per-class matrix, not a single column).
        np.testing.assert_allclose(result.y_prob.sum(axis=1), np.ones(n), atol=1e-6)


class TestMulticlassProbaMissingClass(unittest.TestCase):
    def test_missing_class_computes_and_aligns(self):
        # Model trained on 4 classes, dataset slice missing class 3.
        model, dataset = _fit_model_and_dataset(n_classes=4, drop_class=3)
        result = multiclass_proba(model, dataset, "GINITable")

        # Full training class list drives column alignment / binarization ...
        np.testing.assert_array_equal(result.class_list, [0, 1, 2, 3])
        self.assertEqual(result.y_prob.shape[1], 4)
        self.assertEqual(result.y_bin.shape[1], 4)

        # ... but only the present classes are surfaced for per-class output.
        self.assertEqual([int(c) for c in result.classes_present], [0, 1, 2])
        self.assertEqual(result.present_indices, [0, 1, 2])
        # The dropped class's column carries no positives.
        self.assertEqual(int(result.y_bin[:, 3].sum()), 0)


class TestMulticlassProbaSkips(unittest.TestCase):
    def test_no_predict_proba_skips(self):
        model, dataset = _fit_model_and_dataset(n_classes=3)
        model.model.predict_proba = None
        with self.assertRaises(SkipTestError):
            multiclass_proba(model, dataset, "ROC Curve")

    def test_raising_predict_proba_skips(self):
        model, dataset = _fit_model_and_dataset(n_classes=3)

        def _boom(_X):
            raise RuntimeError("no probabilities here")

        model.model.predict_proba = _boom
        with self.assertRaises(SkipTestError):
            multiclass_proba(model, dataset, "ROC Curve")

    def test_wrong_width_matrix_skips(self):
        model, dataset = _fit_model_and_dataset(n_classes=3)
        # A matrix with fewer columns than the class list must skip, not crash.
        model.model.predict_proba = lambda X: np.zeros((len(X), 2))
        with self.assertRaises(SkipTestError):
            multiclass_proba(model, dataset, "GINITable")

    def test_proba_matrix_wrong_width_skips(self):
        # proba_matrix guards width directly against the supplied class list.
        model, dataset = _fit_model_and_dataset(n_classes=3)
        with self.assertRaises(SkipTestError):
            proba_matrix(model, dataset, np.array([0, 1, 2, 3]), "PSI")

    def test_binary_model_on_multiclass_dataset_skips(self):
        # Model trained on only 2 classes but evaluated against a 3-class dataset:
        # the width check alone would pass (2 == 2), but one-vs-rest multiclass
        # metrics don't apply to a binary model, so this must skip rather than
        # crash when label_binarize's single-column output gets indexed as [:, 1].
        model, dataset = _fit_model_and_dataset(n_classes=3)
        model.model.classes_ = np.array([0, 1])
        model.model.predict_proba = lambda X: np.tile([0.5, 0.5], (len(X), 1))
        with self.assertRaises(SkipTestError):
            multiclass_proba(model, dataset, "ROC Curve")

    def test_unseen_label_in_dataset_skips(self):
        # Model trained on classes {0, 1, 2}; dataset's target column includes a
        # label (5) the model never saw. label_binarize would silently give those
        # rows an all-zero column instead of raising, so this must skip explicitly.
        model, dataset = _fit_model_and_dataset(n_classes=3)
        dataset._df.loc[dataset._df.index[:5], dataset.target_column] = 5
        with self.assertRaises(SkipTestError):
            multiclass_proba(model, dataset, "GINITable")


if __name__ == "__main__":
    unittest.main()
