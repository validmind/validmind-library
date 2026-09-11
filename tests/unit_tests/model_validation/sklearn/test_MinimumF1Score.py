import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import validmind as vm
from validmind.tests.model_validation.sklearn.MinimumF1Score import MinimumF1Score


def _dataset_with_predictions(input_id, y_true, y_pred):
    """Build a VMDataset whose predictions are injected verbatim.

    A fitted model is only needed to construct a valid VMModel; the predictions
    are supplied via ``prediction_values`` so the model's ``predict`` is never
    called and the true/predicted label sets can be controlled exactly.
    """
    df = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, 1.0, len(y_true)),
            "f2": np.linspace(1.0, -1.0, len(y_true)),
            "target": y_true,
        }
    )
    dataset = vm.init_dataset(
        input_id=input_id, dataset=df, target_column="target", __log=False
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(df[["f1", "f2"]].to_numpy(), np.array(y_true))
    vm_model = vm.init_model(input_id=f"{input_id}_model", model=model, __log=False)

    dataset.assign_predictions(model=vm_model, prediction_values=y_pred)
    return dataset, vm_model


class TestMinimumF1Score(unittest.TestCase):
    def test_predicted_class_absent_from_true_labels(self):
        # Regression test for ZD-704. This split's true labels are binary ({0, 1}),
        # but the model predicts a third class. scikit-learn's f1_score derives the
        # target type from the union of y_true and y_pred, so deciding the averaging
        # mode from y_true alone selects average="binary" and raises
        # "Target is multiclass but average='binary'". The test must instead detect
        # the multiclass label space and use macro averaging.
        y_true = [0, 1, 1, 0, 1, 0, 1, 0]
        y_pred = [0, 1, 2, 0, 1, 0, 2, 0]  # class 2 never appears in y_true
        dataset, model = _dataset_with_predictions("f1_multiclass_pred", y_true, y_pred)

        result = MinimumF1Score(dataset, model, min_threshold=0.5)

        score = result[0][0]["Score"]
        expected = f1_score(np.array(y_true), np.array(y_pred), average="macro")
        # Matching the macro value confirms the multiclass branch was taken (the
        # binary branch would have raised rather than returned a number).
        self.assertAlmostEqual(score, expected)
        self.assertEqual(result[1], score > 0.5)

    def test_binary_uses_binary_average(self):
        # A genuinely binary problem (true and predicted labels both within {0, 1})
        # must still use sklearn's default binary averaging and be unaffected by the
        # fix.
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        dataset, model = _dataset_with_predictions("f1_binary", y_true, y_pred)

        result = MinimumF1Score(dataset, model, min_threshold=0.5)

        score = result[0][0]["Score"]
        expected = f1_score(np.array(y_true), np.array(y_pred))  # binary default
        self.assertAlmostEqual(score, expected)
        self.assertEqual(result[1], score > 0.5)


if __name__ == "__main__":
    unittest.main()
