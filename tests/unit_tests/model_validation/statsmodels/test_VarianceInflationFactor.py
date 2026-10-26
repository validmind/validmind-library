import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.model_validation.statsmodels.VarianceInflationFactor import (
    VarianceInflationFactor,
)


def _data(seed=0, n=500):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x1 + rng.normal(0, 0.05, n),  # near-duplicate of x1
            "x3": rng.normal(0, 1, n),
        }
    )
    df["y"] = 2 * df.x1 + df.x3 + rng.normal(0, 1, n)
    return df


def _model(fit, input_id="m"):
    return vm.init_model(fit, input_id=input_id, __log=False)


class TestVarianceInflationFactor(unittest.TestCase):
    def setUp(self):
        self.df = _data()

    def test_flags_collinear_regressors(self):
        model = _model(smf.ols("y ~ x1 + x2 + x3", self.df).fit())
        tables, fig, passed, _ = VarianceInflationFactor(model)
        vif = tables["Variance Inflation Factor"].set_index("Feature")

        self.assertFalse(passed)
        self.assertEqual(vif.loc["x1", "Status"], "Severe")
        self.assertEqual(vif.loc["x2", "Status"], "Severe")
        self.assertEqual(vif.loc["x3", "Status"], "OK")
        self.assertNotIn("Intercept", vif.index)
        self.assertIsInstance(fig, go.Figure)

    def test_passes_without_collinearity(self):
        model = _model(smf.ols("y ~ x1 + x3", self.df).fit())
        tables, _, passed, _ = VarianceInflationFactor(model)

        self.assertTrue(passed)
        self.assertTrue((tables["Variance Inflation Factor"]["VIF"] < 2).all())
        self.assertEqual(tables["Condition Number"]["Pass/Fail"].iloc[0], "Pass")

    def test_thresholds_are_parameters(self):
        model = _model(smf.ols("y ~ x1 + x3", self.df).fit())
        _, _, passed, _ = VarianceInflationFactor(
            model, vif_threshold=1.0, moderate_vif_threshold=0.5
        )
        self.assertFalse(passed)

    def test_array_api_matches_formula_api(self):
        formula = _model(smf.ols("y ~ x1 + x2 + x3", self.df).fit())
        array = _model(
            sm.OLS(self.df.y, sm.add_constant(self.df[["x1", "x2", "x3"]])).fit()
        )
        vif_formula = VarianceInflationFactor(formula)[0]["Variance Inflation Factor"]
        vif_array = VarianceInflationFactor(array)[0]["Variance Inflation Factor"]

        self.assertNotIn("const", vif_array["Feature"].tolist())
        np.testing.assert_allclose(vif_formula["VIF"], vif_array["VIF"])

    def test_skips_non_statsmodels_models(self):
        sk = LinearRegression().fit(self.df[["x1", "x3"]], self.df.y)
        with self.assertRaises(SkipTestError):
            VarianceInflationFactor(_model(sk))

    def test_status_and_pass_fail_use_the_value_shown(self):
        fit = smf.ols("y ~ x1 + x2 + x3", self.df).fit()
        raw = {
            name: variance_inflation_factor(fit.model.exog, i)
            for i, name in enumerate(fit.model.exog_names)
            if name != "Intercept"
        }
        # a threshold the raw VIF is just below and the shown (rounded) VIF reaches
        name, threshold = next(
            (n, round(v, 2)) for n, v in raw.items() if round(v, 2) > v
        )
        tables, _, passed, _ = VarianceInflationFactor(
            _model(fit), vif_threshold=threshold, moderate_vif_threshold=threshold / 2
        )
        row = tables["Variance Inflation Factor"].set_index("Feature").loc[name]

        self.assertEqual(row["VIF"], threshold)
        self.assertEqual(row["Status"], "Severe")
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
