import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind.tests.model_validation.ScenarioAnalysis import ScenarioAnalysis


class TestScenarioAnalysis(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 500
        self.df = pd.DataFrame({"a": rng.uniform(1, 2, n), "b": rng.uniform(0, 1, n)})
        self.df["y"] = 10 + 2 * self.df.a - 5 * self.df.b
        fit = LinearRegression().fit(self.df[["a", "b"]], self.df.y)
        self.model = vm.init_model(fit, input_id="m", __log=False)
        self.dataset = vm.init_dataset(self.df, input_id="d", target_column="y", __log=False)
        self.dataset.assign_predictions(self.model)

    def _run(self, scenarios, **kwargs):
        table, fig, _ = ScenarioAnalysis(self.dataset, self.model, scenarios, **kwargs)
        return table.set_index("Scenario"), fig

    def test_operations_have_exact_linear_effects(self):
        n = len(self.df)
        table, fig = self._run(
            {
                "double a": {"a": {"multiply": 2}},
                "b up": {"b": {"add": 0.1}},
                "b zero": {"b": {"set": 0}},
            }
        )

        self.assertAlmostEqual(table.loc["double a", "Change in Total"], 2 * self.df.a.sum(), places=4)
        self.assertAlmostEqual(table.loc["b up", "Change in Total"], -5 * 0.1 * n, places=4)
        self.assertAlmostEqual(table.loc["b zero", "Change in Total"], 5 * self.df.b.sum(), places=4)
        self.assertEqual(table.loc["Baseline", "Change in Total"], 0)
        self.assertIsInstance(fig, go.Figure)

    def test_joint_changes_in_one_scenario(self):
        table, _ = self._run({"both": {"a": {"add": 1}, "b": {"add": 1}}})
        self.assertAlmostEqual(table.loc["both", "Mean Prediction"] - table.loc["Baseline", "Mean Prediction"], -3, places=4)
        self.assertEqual(table.loc["both", "Changes"], "a add 1; b add 1")

    def test_reports_share_below_lower_bound(self):
        table, _ = self._run({"crash": {"b": {"set": 10}}}, lower_bound=0)
        self.assertEqual(table.loc["Baseline", "Share Below 0"], 0)
        self.assertEqual(table.loc["crash", "Share Below 0"], 1)

    def test_does_not_modify_the_dataset(self):
        before = self.dataset.df.copy()
        self._run({"double a": {"a": {"multiply": 2}}})
        pd.testing.assert_frame_equal(self.dataset.df, before)

    def test_invalid_scenarios_raise(self):
        for scenarios, message in [
            ({"s": {"missing": {"add": 1}}}, "unknown feature"),
            ({"s": {"a": {"divide": 2}}}, "unknown operation"),
            ({"s": {"a": 2}}, "exactly one of"),
            ({"s": {"a": {"add": 1, "set": 2}}}, "exactly one of"),
            ({"Baseline": {"a": {"add": 1}}}, "reserved"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                self._run(scenarios)


if __name__ == "__main__":
    unittest.main()
