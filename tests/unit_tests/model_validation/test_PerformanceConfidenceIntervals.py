import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.PerformanceConfidenceIntervals import (
    PerformanceConfidenceIntervals,
)


class TestPerformanceConfidenceIntervals(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 1000
        df = pd.DataFrame({"x": rng.normal(0, 1, n)})
        df["y"] = 2 * df.x + rng.normal(0, 1, n)
        self.dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)

        shared = rng.normal(0, 1, n)
        self.models = {}
        for name, noise in [
            ("good", 0.3 * shared),
            ("bad", 0.3 * shared + rng.normal(0, 1.0, n)),  # clearly worse
            ("twin", 0.3 * shared + rng.normal(0, 0.01, n)),  # same as good
        ]:
            model = vm.init_model(input_id=name, predict_fn=lambda row: 0.0, __log=False)
            self.dataset.assign_predictions(
                model, prediction_values=(2 * df.x + noise).to_numpy()
            )
            self.models[name] = model

    def test_single_model_gets_intervals(self):
        tables, fig, _ = PerformanceConfidenceIntervals(
            self.dataset, [self.models["good"]], n_bootstrap=200
        )
        perf = tables["Performance with Confidence Intervals"].set_index("Metric")

        self.assertEqual(list(tables), ["Performance with Confidence Intervals"])
        for metric in ["R-Squared", "RMSE", "MAE", "Mean Bias"]:
            row = perf.loc[metric]
            self.assertLessEqual(row["95% CI Lower"], row["Estimate"])
            self.assertGreaterEqual(row["95% CI Upper"], row["Estimate"])
        self.assertIsInstance(fig, go.Figure)

    def test_difference_identifies_the_better_model(self):
        tables, _, _ = PerformanceConfidenceIntervals(
            self.dataset, [self.models["bad"], self.models["good"]], n_bootstrap=300
        )
        diff = tables["Difference vs bad"].set_index("Metric")

        self.assertTrue(diff["CI Excludes Zero"].all())
        self.assertTrue((diff["Better Model"] == "good").all())
        self.assertLess(diff.loc["MAE", "Difference"], 0)

    def test_no_difference_between_equivalent_models(self):
        tables, _, _ = PerformanceConfidenceIntervals(
            self.dataset, [self.models["good"], self.models["twin"]], n_bootstrap=300
        )
        diff = tables["Difference vs good"]
        self.assertTrue((diff["Better Model"] == "No significant difference").all())

    def test_deterministic_for_a_seed(self):
        args = (self.dataset, [self.models["good"], self.models["bad"]])
        a = PerformanceConfidenceIntervals(*args, n_bootstrap=50, seed=3)[0]
        b = PerformanceConfidenceIntervals(*args, n_bootstrap=50, seed=3)[0]
        for key in a:
            pd.testing.assert_frame_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
