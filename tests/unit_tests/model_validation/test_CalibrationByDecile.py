import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.CalibrationByDecile import CalibrationByDecile


def _dataset(prediction_fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    df["y"] = df.x + rng.normal(0, 1, n)
    model = vm.init_model(input_id="m", predict_fn=lambda row: 0.0, __log=False)
    dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
    dataset.assign_predictions(model, prediction_values=prediction_fn(df.x).to_numpy())
    return dataset, model


class TestCalibrationByDecile(unittest.TestCase):
    def test_calibrated_model_passes(self):
        table, fig, passed, _ = CalibrationByDecile(*_dataset(lambda x: x))
        self.assertTrue(passed)
        self.assertEqual(len(table), 10)
        self.assertIsInstance(fig, go.Figure)

    def test_too_flat_model_fails_at_both_ends(self):
        # predictions shrunk towards the mean: under-predicts the top, over-predicts the bottom
        table, _, passed, _ = CalibrationByDecile(*_dataset(lambda x: 5 + 0.6 * (x - 5)))
        self.assertFalse(passed)
        self.assertEqual(table["Result"].iloc[0], "Miscalibrated")
        self.assertEqual(table["Result"].iloc[-1], "Miscalibrated")
        self.assertLess(table["Actual − Predicted"].iloc[0], 0)
        self.assertGreater(table["Actual − Predicted"].iloc[-1], 0)

    def test_number_of_bins_is_a_parameter(self):
        table = CalibrationByDecile(*_dataset(lambda x: x), n_bins=5)[0]
        self.assertEqual(len(table), 5)
        self.assertEqual(table["Count"].sum(), 5000)


if __name__ == "__main__":
    unittest.main()
