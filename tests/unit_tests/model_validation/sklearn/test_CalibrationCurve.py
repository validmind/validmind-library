import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.model_validation.sklearn.CalibrationCurve import CalibrationCurve


def _dataset_and_model(labels, seed):
    n = 200
    rng = np.random.RandomState(seed)
    y = rng.choice(labels, size=n)
    df = pd.DataFrame(
        {"f1": np.linspace(0.0, 10.0, n), "f2": rng.randn(n), "target": y}
    )
    ds = vm.init_dataset(
        input_id=f"cc_{seed}", dataset=df, target_column="target", __log=False
    )
    est = LogisticRegression(max_iter=1000).fit(df[["f1", "f2"]].to_numpy(), y)
    model = vm.init_model(input_id=f"cc_{seed}_m", model=est, __log=False)
    ds.assign_predictions(model=model)
    return ds, model


class TestCalibrationCurve(unittest.TestCase):
    def test_multiclass_skips_cleanly(self):
        # sklearn's calibration_curve is binary-only and raised a cryptic
        # "pos_label is not specified" error on multiclass; it must skip instead.
        ds, model = _dataset_and_model([0, 1, 2, 3, 4], seed=1)
        with self.assertRaises(SkipTestError):
            CalibrationCurve(model=model, dataset=ds)

    def test_binary_runs(self):
        ds, model = _dataset_and_model([0, 1], seed=2)
        result = CalibrationCurve(model=model, dataset=ds)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
