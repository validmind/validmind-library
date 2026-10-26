import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import validmind as vm
from validmind.tests.model_validation.statsmodels.CoefficientStability import (
    CoefficientStability,
)


def _model(fit):
    return vm.init_model(fit, input_id="m", __log=False)


class TestCoefficientStability(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        self.df = pd.DataFrame({"x": rng.normal(0, 1, n), "noise": rng.normal(0, 1, n)})
        self.df["y"] = 2 * self.df.x + rng.normal(0, 1, n)  # "noise" has no effect

    def test_strong_effect_is_stable(self):
        table, passed, _ = CoefficientStability(
            _model(smf.ols("y ~ x", self.df).fit()), n_bootstrap=200
        )
        row = table.set_index("Coefficient").loc["x"]

        self.assertTrue(passed)
        self.assertEqual(row["Sign Stability"], 1.0)
        self.assertLess(row["Bootstrap 95% CI Lower"], 2)
        self.assertGreater(row["Bootstrap 95% CI Upper"], 2)

    def test_no_effect_is_unstable(self):
        table, passed, _ = CoefficientStability(
            _model(smf.ols("y ~ x + noise", self.df).fit()), n_bootstrap=200
        )
        rows = table.set_index("Coefficient")

        self.assertFalse(passed)
        self.assertEqual(rows.loc["noise", "Pass/Fail"], "Fail")
        self.assertEqual(rows.loc["x", "Pass/Fail"], "Pass")
        self.assertEqual(rows.loc["Intercept", "Pass/Fail"], "N/A")

    def test_deterministic_for_a_seed(self):
        model = _model(smf.ols("y ~ x + noise", self.df).fit())
        a = CoefficientStability(model, n_bootstrap=50, seed=7)[0]
        b = CoefficientStability(model, n_bootstrap=50, seed=7)[0]
        pd.testing.assert_frame_equal(a, b)

    def test_array_api(self):
        X = sm.add_constant(self.df[["x", "noise"]])
        table, passed, raw = CoefficientStability(
            _model(sm.OLS(self.df.y, X).fit()), n_bootstrap=100
        )
        self.assertFalse(passed)
        self.assertEqual(table.set_index("Coefficient").loc["const", "Pass/Fail"], "N/A")
        self.assertEqual(raw.bootstrap_coefficients.shape, (100, 3))


if __name__ == "__main__":
    unittest.main()
