import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import validmind as vm
from validmind.tests.model_validation.sklearn.OverfitDiagnosis import OverfitDiagnosis


def _pair(labels, seed=0, n=200, classification=True):
    """Build a train/test VMDataset pair sharing one fitted VMModel."""

    def build(input_id, s, model=None):
        r = np.random.RandomState(s)
        f1 = np.linspace(0.0, 10.0, n)
        f2 = r.randn(n)
        if classification:
            y = r.choice(labels, size=n)
            # ensure the lowest feature bin holds only a subset of the classes
            y[:20] = r.choice(labels[:2], size=20)
        else:
            y = 3.0 * f1 + 2.0 * f2 + r.randn(n)
        df = pd.DataFrame({"f1": f1, "f2": f2, "target": y})
        ds = vm.init_dataset(
            input_id=input_id, dataset=df, target_column="target", __log=False
        )
        if model is None:
            est = (
                LogisticRegression(max_iter=2000)
                if classification
                else LinearRegression()
            )
            est.fit(df[["f1", "f2"]].to_numpy(), y)
            model = vm.init_model(input_id=f"{input_id}_m", model=est, __log=False)
        ds.assign_predictions(model=model)
        return ds, model

    train, model = build(f"of_{seed}_train", seed)
    test, _ = build(f"of_{seed}_test", seed + 1, model=model)
    return train, test, model


class TestOverfitDiagnosis(unittest.TestCase):
    def test_multiclass_default_auc_runs(self):
        # Regression test (ZD-704 family): default metric is "auc"; multiclass
        # previously raised "multi_class must be in ('ovo', 'ovr')".
        train, test, model = _pair([0, 1, 2, 3, 4], seed=1)
        result = OverfitDiagnosis(model=model, datasets=[train, test])
        self.assertIsNotNone(result)

    def test_multiclass_f1_runs(self):
        # metric="f1" previously raised "Target is multiclass but average='binary'".
        train, test, model = _pair([0, 1, 2, 3, 4], seed=2)
        result = OverfitDiagnosis(model=model, datasets=[train, test], metric="f1")
        self.assertIsNotNone(result)

    def test_binary_non_standard_labels_run(self):
        # Binary target encoded as {0, 4}: default auc and f1 must both run.
        train, test, model = _pair([0, 4], seed=3)
        self.assertIsNotNone(OverfitDiagnosis(model=model, datasets=[train, test]))
        self.assertIsNotNone(
            OverfitDiagnosis(model=model, datasets=[train, test], metric="f1")
        )

    def test_standard_binary_runs(self):
        train, test, model = _pair([0, 1], seed=4)
        self.assertIsNotNone(OverfitDiagnosis(model=model, datasets=[train, test]))

    def test_regression_runs(self):
        train, test, model = _pair(None, seed=5, classification=False)
        self.assertIsNotNone(OverfitDiagnosis(model=model, datasets=[train, test]))


if __name__ == "__main__":
    unittest.main()
