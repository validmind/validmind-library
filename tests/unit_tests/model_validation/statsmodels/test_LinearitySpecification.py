import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf

import validmind as vm
from validmind.tests.model_validation.statsmodels.LinearitySpecification import (
    LinearitySpecification,
)


def _model(fit):
    return vm.init_model(fit, input_id="m", __log=False)


class TestLinearitySpecification(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 500
        self.df = pd.DataFrame(
            {"x": rng.uniform(0, 3, n), "z": rng.normal(0, 1, n), "flag": rng.integers(0, 2, n)}
        )
        noise = rng.normal(0, 0.5, n)
        self.df["y_linear"] = 1 + 2 * self.df.x + self.df.z + self.df.flag + noise
        self.df["y_curved"] = 1 + 2 * self.df.x**2 + self.df.z + self.df.flag + noise

    def test_detects_missing_curvature(self):
        model = _model(smf.ols("y_curved ~ x + z + flag", self.df).fit())
        table, fig, passed, raw = LinearitySpecification(model)

        self.assertFalse(passed)
        self.assertEqual(table["Pass/Fail"].iloc[0], "Fail")
        self.assertIsInstance(fig, go.Figure)

    def test_passes_correct_specification(self):
        model = _model(smf.ols("y_linear ~ x + z + flag", self.df).fit())
        table, _, passed, _ = LinearitySpecification(model)

        self.assertTrue(passed)
        self.assertGreaterEqual(table["p-value"].iloc[0], 0.05)

    def test_plots_continuous_regressors_only(self):
        model = _model(smf.ols("y_curved ~ x + z + flag", self.df).fit())
        _, fig, _, raw = LinearitySpecification(model)

        titles = [a.text for a in fig.layout.annotations]
        self.assertEqual(titles, ["x", "z"])  # no intercept, no binary flag
        self.assertEqual(set(raw.lowess), {"x", "z"})

    def test_array_api(self):
        X = sm.add_constant(self.df[["x", "z", "flag"]])
        model = _model(sm.OLS(self.df.y_curved, X).fit())
        _, fig, passed, _ = LinearitySpecification(model)

        self.assertFalse(passed)
        self.assertEqual([a.text for a in fig.layout.annotations], ["x", "z"])

    def test_samples_points_but_not_smoother(self):
        model = _model(smf.ols("y_linear ~ x + z + flag", self.df).fit())
        _, fig, _, raw = LinearitySpecification(model, max_points=100)

        self.assertEqual(len(fig.data[0].x), 100)
        self.assertEqual(len(raw.lowess["x"]), len(self.df))


if __name__ == "__main__":
    unittest.main()
