import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.sklearn.ConfusionMatrix import ConfusionMatrix


def _build(n_classes=2):
    """Fit a logistic regression on ``n_classes`` and return (dataset, model)."""
    X, y = make_classification(
        n_samples=400,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=7,
    )
    model = LogisticRegression(max_iter=1000).fit(X, y)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    vm_model = vm.init_model(input_id="cm_model", model=model, __log=False)
    vm_dataset = vm.init_dataset(
        input_id="cm_dataset", dataset=df, target_column="target", __log=False
    )
    vm_dataset.assign_predictions(vm_model)

    return vm_dataset, vm_model


class TestConfusionMatrix(unittest.TestCase):
    def test_returns_heatmap_and_raw_data(self):
        dataset, model = _build()
        fig, raw_data = ConfusionMatrix(dataset, model)

        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(raw_data, RawData)
        self.assertEqual(fig.data[0].type, "heatmap")

    def test_z_matches_sklearn_confusion_matrix(self):
        dataset, model = _build()
        fig, _ = ConfusionMatrix(dataset, model)

        y_prob = dataset.y_prob(model)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        y_true = dataset.y.astype(y_pred.dtype)
        labels = sorted(np.unique(y_true).tolist())

        expected = confusion_matrix(y_true, y_pred, labels=labels)
        np.testing.assert_array_equal(np.asarray(fig.data[0].z), expected)

    def test_binary_labels_are_not_transposed(self):
        """TN must sit top-left and TP bottom-right.

        The cell text is the only thing naming which quadrant is which, so a
        row/column flip in the heatmap would be silent without this assertion.
        """
        dataset, model = _build()
        fig, _ = ConfusionMatrix(dataset, model)

        text = fig.data[0].text
        self.assertIn("True Negatives", text[0][0])
        self.assertIn("False Positives", text[0][1])
        self.assertIn("False Negatives", text[1][0])
        self.assertIn("True Positives", text[1][1])
        self.assertEqual(fig.data[0].texttemplate, "%{text}")

    def test_multiclass_annotates_raw_counts(self):
        dataset, model = _build(n_classes=3)
        fig, _ = ConfusionMatrix(dataset, model)

        z = np.asarray(fig.data[0].z)
        self.assertEqual(z.shape, (3, 3))

        # Without the binary TN/FP/FN/TP labels, every cell shows its own count.
        text = np.asarray(fig.data[0].text)
        np.testing.assert_array_equal(text, z.astype(str))


if __name__ == "__main__":
    unittest.main()
