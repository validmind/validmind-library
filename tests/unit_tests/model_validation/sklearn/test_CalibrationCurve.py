import unittest

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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

    def test_binary_non_standard_encoding_runs(self):
        # Follow-up to the #534 review: a two-class target encoded as {0, 4} passes
        # the multiclass guard but previously hit sklearn's "pos_label is not
        # specified" error. Resolving the positive label to the largest value fixes
        # it.
        ds, model = _dataset_and_model([0, 4], seed=3)
        result = CalibrationCurve(model=model, dataset=ds)
        self.assertIsNotNone(result)

    def test_binary_string_labels_run(self):
        # String labels ("bad"/"good") are the same failure family and must not
        # raise.
        ds, model = _dataset_and_model(["bad", "good"], seed=4)
        result = CalibrationCurve(model=model, dataset=ds)
        self.assertIsNotNone(result)

    def test_standard_binary_curve_unchanged(self):
        # Backward compatibility: for {0, 1} the raw curve must be byte-identical to
        # sklearn's default (which resolves pos_label to 1) so existing results are
        # unaffected.
        ds, model = _dataset_and_model([0, 1], seed=2)
        _, raw_data = CalibrationCurve(model=model, dataset=ds)
        expected_true, expected_pred = calibration_curve(
            ds.y, ds.y_prob(model), n_bins=10
        )
        np.testing.assert_array_equal(raw_data.observed_frequency, expected_true)
        np.testing.assert_array_equal(
            raw_data.mean_predicted_probability, expected_pred
        )


if __name__ == "__main__":
    unittest.main()
