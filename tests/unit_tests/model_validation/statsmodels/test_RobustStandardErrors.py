import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import validmind as vm
from validmind.tests.model_validation.statsmodels.RobustStandardErrors import (
    RobustStandardErrors,
)


def _model(fit):
    return vm.init_model(fit, input_id="m", __log=False)


def _data(heteroscedastic, seed=1, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    if heteroscedastic:
        noise = noise * (0.2 + 2 * x**2)  # variance grows at the extremes of x
    return pd.DataFrame({"x": x, "y": 0.25 * x + noise})


class TestRobustStandardErrors(unittest.TestCase):
    def test_flags_significance_that_depends_on_constant_variance(self):
        table, passed, _ = RobustStandardErrors(
            _model(smf.ols("y ~ x", _data(heteroscedastic=True)).fit())
        )
        row = table.set_index("Coefficient").loc["x"]

        self.assertFalse(passed)
        self.assertTrue(row["Significant (OLS)"])
        self.assertFalse(row["Significant (HC3)"])
        self.assertGreater(row["Ratio (Robust / OLS)"], 1.5)

    def test_passes_with_constant_variance(self):
        table, passed, _ = RobustStandardErrors(
            _model(smf.ols("y ~ x", _data(heteroscedastic=False)).fit())
        )
        self.assertTrue(passed)
        self.assertTrue(table["Ratio (Robust / OLS)"].between(0.8, 1.25).all())

    def test_cov_type_is_a_parameter(self):
        table, _, _ = RobustStandardErrors(
            _model(smf.ols("y ~ x", _data(heteroscedastic=True)).fit()), cov_type="HC1"
        )
        self.assertIn("HC1 Std. Error", table.columns)

    def test_array_api(self):
        df = _data(heteroscedastic=True)
        _, passed, _ = RobustStandardErrors(
            _model(sm.OLS(df.y, sm.add_constant(df[["x"]])).fit())
        )
        self.assertFalse(passed)

    def test_model_fitted_with_robust_errors_is_still_compared_to_ols(self):
        df = _data(heteroscedastic=True)
        plain = RobustStandardErrors(_model(smf.ols("y ~ x", df).fit()))
        robust_fit = RobustStandardErrors(
            _model(smf.ols("y ~ x", df).fit(cov_type="HC3"))
        )

        self.assertFalse(robust_fit[1])
        pd.testing.assert_frame_equal(plain[0], robust_fit[0])


if __name__ == "__main__":
    unittest.main()
