import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf

import validmind as vm
from validmind.tests.model_validation.statsmodels.InfluentialObservations import (
    InfluentialObservations,
)


def _model(fit):
    return vm.init_model(fit, input_id="m", __log=False)


class TestInfluentialObservations(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        self.df = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n)})
        self.df["y"] = 3 * self.df.x + self.df.z + rng.normal(0, 1, n)
        self.df.index = [f"row{i}" for i in range(n)]

        self.bad = self.df.copy()
        self.bad.loc["row7", "x"] = 40.0  # data entry error: extreme x, ordinary y

    def test_passes_clean_data(self):
        tables, fig, passed, _ = InfluentialObservations(
            _model(smf.ols("y ~ x + z", self.df).fit())
        )
        self.assertTrue(passed)
        self.assertIn("Result", tables["Coefficient Impact of Severe Observations"])
        self.assertIsInstance(fig, go.Figure)

    def test_flags_the_planted_observation_by_its_label(self):
        tables, _, passed, _ = InfluentialObservations(
            _model(smf.ols("y ~ x + z", self.bad).fit()), top_n=5
        )
        top = tables["Top 5 Observations by Cook's Distance"]

        self.assertFalse(passed)
        self.assertEqual(top["Observation"].iloc[0], "row7")
        self.assertGreater(top["Cook's Distance"].iloc[0], 1)

    def test_reports_coefficient_impact(self):
        tables, _, _, _ = InfluentialObservations(
            _model(smf.ols("y ~ x + z", self.bad).fit())
        )
        impact = tables["Coefficient Impact of Severe Observations"].set_index(
            "Coefficient"
        )
        # without the error the slope recovers towards the true value of 3
        self.assertAlmostEqual(impact.loc["x", "Without 1 Severe"], 3, delta=0.3)
        self.assertGreater(abs(impact.loc["x", "Change"]), 1)

    def test_default_thresholds_follow_conventions(self):
        fit = smf.ols("y ~ x + z", self.df).fit()
        summary = InfluentialObservations(_model(fit))[0]["Influence Summary"]
        n, p = int(fit.nobs), len(fit.params)

        self.assertAlmostEqual(summary["Threshold"].iloc[0], round(4 / n, 6))
        self.assertAlmostEqual(summary["Threshold"].iloc[2], round(2 * p / n, 6))

    def test_array_api(self):
        X = sm.add_constant(self.bad[["x", "z"]])
        _, _, passed, _ = InfluentialObservations(_model(sm.OLS(self.bad.y, X).fit()))
        self.assertFalse(passed)


class TestExternallyStudentizedResiduals(unittest.TestCase):
    def test_matches_statsmodels(self):
        from validmind.tests.model_validation.statsmodels.InfluentialObservations import (
            _externally_studentized,
        )

        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 120), "z": rng.normal(0, 1, 120)})
        df["y"] = df.x - 2 * df.z + rng.standard_t(3, 120)
        fit = smf.ols("y ~ x + z", df).fit()
        influence = fit.get_influence()

        np.testing.assert_allclose(
            _externally_studentized(fit, influence.hat_matrix_diag),
            influence.resid_studentized_external,
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
