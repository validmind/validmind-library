import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.RegressionWeakspots import RegressionWeakspots


def _dataset(noisy_scale, n=4000, seed=0):
    """Noise is `noisy_scale` times larger for group == 'noisy'; no bias anywhere."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "group": rng.choice(["noisy", "a", "b", "c"], n)})
    scale = np.where(df.group == "noisy", noisy_scale, 1.0)
    prediction = 2 * df.x
    df["y"] = prediction + rng.normal(0, 1, n) * scale
    model = vm.init_model(input_id="m", predict_fn=lambda row: 0.0, __log=False)
    dataset = vm.init_dataset(
        df, input_id="d", target_column="y", feature_columns=["x"],
        extra_columns={"group": None}, __log=False,
    )
    dataset.assign_predictions(model, prediction_values=prediction.to_numpy())
    return dataset, model


class TestRegressionWeakspots(unittest.TestCase):
    def test_flags_the_noisy_segment(self):
        table, fig, passed, _ = RegressionWeakspots(*_dataset(noisy_scale=3.0))
        weak = table[table.Result == "Weak spot"]

        self.assertFalse(passed)
        self.assertEqual(list(weak["Segment"]), ["noisy"])
        self.assertGreater(weak["MAE Ratio to Overall"].iloc[0], 1.25)
        self.assertIsInstance(fig, go.Figure)

    def test_quiet_with_uniform_error(self):
        _, _, passed, _ = RegressionWeakspots(*_dataset(noisy_scale=1.0))
        self.assertTrue(passed)

    def test_missing_values_in_a_binned_column_do_not_crash(self):
        dataset, model = _dataset(noisy_scale=1.0)
        dataset.add_extra_column("score", np.r_[np.full(10, np.nan), np.arange(3990.0)])
        table = RegressionWeakspots(dataset, model, segment_columns=["score"])[0]

        self.assertEqual(table["Segment"].iloc[-1], "Missing")

    def test_requires_a_material_difference(self):
        # 1.15x noise is statistically clear on 1,000 rows but below the 1.25x ratio
        table, _, passed, _ = RegressionWeakspots(*_dataset(noisy_scale=1.15))
        self.assertTrue(passed)
        _, _, passed_strict, _ = RegressionWeakspots(*_dataset(noisy_scale=1.15), max_error_ratio=1.05)
        self.assertFalse(passed_strict)


if __name__ == "__main__":
    unittest.main()
