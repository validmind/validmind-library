import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.PredictionRange import PredictionRange


class TestPredictionRange(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
        self.model = vm.init_model(input_id="m", predict_fn=lambda row: 0.0, __log=False)
        self.dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
        # predictions -2 .. 7
        self.dataset.assign_predictions(self.model, prediction_values=np.arange(10.0) - 2)

    def test_counts_predictions_below_the_lower_bound(self):
        table, fig, passed, _ = PredictionRange(self.dataset, self.model, lower_bound=0)
        lower = table.set_index("Bound").loc["Lower"]

        self.assertFalse(passed)
        self.assertEqual(lower["Predictions Beyond"], 2)
        self.assertAlmostEqual(lower["Share Beyond"], 0.2)
        self.assertEqual(lower["Most Extreme Prediction"], -2)
        self.assertIsInstance(fig, go.Figure)

    def test_combines_both_bounds(self):
        table, _, _, _ = PredictionRange(
            self.dataset, self.model, lower_bound=0, upper_bound=6
        )
        total = table.set_index("Bound").loc["Total outside range"]
        self.assertEqual(total["Predictions Beyond"], 3)

    def test_bound_of_zero_is_respected(self):
        table, _, _, _ = PredictionRange(self.dataset, self.model, upper_bound=0)
        self.assertEqual(table.set_index("Bound").loc["Upper", "Predictions Beyond"], 7)

    def test_tolerance_is_a_parameter(self):
        _, _, passed, _ = PredictionRange(
            self.dataset, self.model, lower_bound=0, max_share_outside=0.2
        )
        self.assertTrue(passed)

    def test_requires_a_bound(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            PredictionRange(self.dataset, self.model)


if __name__ == "__main__":
    unittest.main()
