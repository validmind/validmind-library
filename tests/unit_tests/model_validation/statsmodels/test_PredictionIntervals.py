import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.model_validation.statsmodels.PredictionIntervals import (
    PredictionIntervals,
)


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n)})
    df["y"] = 1 + 2 * df.x - df.z + rng.normal(0, 1, n) + shift
    return df


def _dataset(df, model, fit):
    dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
    dataset.assign_predictions(model, prediction_values=np.asarray(fit.predict(df)))
    return dataset


class TestPredictionIntervals(unittest.TestCase):
    def setUp(self):
        self.train = _frame(1000, seed=0)
        self.fit = smf.ols("y ~ x + z", self.train).fit()
        self.model = vm.init_model(self.fit, input_id="m", __log=False)

    def test_well_specified_model_covers_new_data(self):
        holdout = _frame(2000, seed=1)
        tables, coverage_fig, band_fig, passed, _ = PredictionIntervals(
            _dataset(holdout, self.model, self.fit), self.model
        )
        coverage = tables["Prediction Interval Coverage"]["Coverage"].iloc[0]

        self.assertTrue(passed)
        self.assertAlmostEqual(coverage, 0.95, delta=0.015)
        self.assertIsInstance(coverage_fig, go.Figure)
        self.assertIsInstance(band_fig, go.Figure)

    def test_detects_shifted_data_and_its_direction(self):
        shifted = _frame(2000, seed=1, shift=1.5)
        tables, _, _, passed, _ = PredictionIntervals(
            _dataset(shifted, self.model, self.fit), self.model
        )
        row = tables["Prediction Interval Coverage"].iloc[0]

        self.assertFalse(passed)
        self.assertGreater(row["Above Interval"], 10 * row["Below Interval"])

    def test_confidence_level_is_a_parameter(self):
        holdout = _frame(2000, seed=1)
        tables = PredictionIntervals(
            _dataset(holdout, self.model, self.fit), self.model, confidence_level=0.8
        )[0]
        coverage = tables["Prediction Interval Coverage"]["Coverage"].iloc[0]
        self.assertAlmostEqual(coverage, 0.8, delta=0.03)

    def test_array_api(self):
        X = sm.add_constant(self.train[["x", "z"]])
        fit = sm.OLS(self.train.y, X).fit()
        model = vm.init_model(fit, input_id="a", __log=False)
        holdout = _frame(2000, seed=1)
        dataset = vm.init_dataset(holdout, input_id="d", target_column="y", __log=False)
        dataset.assign_predictions(
            model, prediction_values=np.asarray(fit.predict(sm.add_constant(holdout[["x", "z"]])))
        )

        formula_tables = PredictionIntervals(
            _dataset(holdout, self.model, self.fit), self.model
        )[0]
        array_tables = PredictionIntervals(dataset, model)[0]
        pd.testing.assert_frame_equal(
            formula_tables["Prediction Interval Coverage"],
            array_tables["Prediction Interval Coverage"],
        )

    def test_skips_non_statsmodels_models(self):
        sk = LinearRegression().fit(self.train[["x", "z"]], self.train.y)
        model = vm.init_model(sk, input_id="sk", __log=False)
        dataset = vm.init_dataset(self.train, input_id="d", target_column="y", __log=False)
        dataset.assign_predictions(model)
        with self.assertRaises(SkipTestError):
            PredictionIntervals(dataset, model)


if __name__ == "__main__":
    unittest.main()
