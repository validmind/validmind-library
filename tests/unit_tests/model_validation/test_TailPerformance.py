import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.TailPerformance import TailPerformance


class TestTailPerformance(unittest.TestCase):
    def setUp(self):
        y = np.arange(1, 1001, dtype=float)
        df = pd.DataFrame({"x": y, "y": y})
        self.model = vm.init_model(
            input_id="m", predict_fn=lambda row: 0.0, __log=False
        )
        self.dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
        # perfect except the top 10% of actual values, under-predicted by 100
        prediction = np.where(y > 900, y - 100, y)
        self.dataset.assign_predictions(self.model, prediction_values=prediction)

    def test_isolates_error_in_the_top_tail(self):
        table, fig, _ = TailPerformance(self.dataset, self.model)
        rows = table.set_index("Group")

        self.assertEqual(rows.loc["Top 10% by actual", "Mean Residual"], 100)
        self.assertEqual(rows.loc["Bottom 10% by actual", "MAE"], 0)
        self.assertAlmostEqual(rows.loc["All observations", "MAE"], 10)
        self.assertEqual(rows.loc["Top 10% by actual", "MAE Ratio to Overall"], 10)
        self.assertIsInstance(fig, go.Figure)

    def test_tail_share_is_a_parameter(self):
        table = TailPerformance(self.dataset, self.model, tail_share=0.05)[0]
        rows = table.set_index("Group")
        self.assertEqual(rows.loc["Top 5% by actual", "Count"], 50)

    def test_ties_do_not_widen_a_tail(self):
        y = np.r_[np.zeros(300), np.arange(1, 701, dtype=float)]  # 30% zeros
        dataset = vm.init_dataset(
            pd.DataFrame({"x": y, "y": y}), input_id="z", target_column="y", __log=False
        )
        dataset.assign_predictions(self.model, prediction_values=y)
        rows = TailPerformance(dataset, self.model)[0].set_index("Group")

        self.assertEqual(rows.loc["Bottom 10% by actual", "Count"], 100)
        self.assertEqual(rows.loc["Bottom 10% by prediction", "Count"], 100)

    def test_tails_have_the_same_size(self):
        y = np.arange(1, 101, dtype=float)
        dataset = vm.init_dataset(
            pd.DataFrame({"x": y, "y": y}), input_id="h", target_column="y", __log=False
        )
        dataset.assign_predictions(self.model, prediction_values=y)
        rows = TailPerformance(dataset, self.model, tail_share=0.29)[0].set_index(
            "Group"
        )

        self.assertEqual(rows.loc["Top 29% by actual", "Count"], 29)
        self.assertEqual(rows.loc["Bottom 29% by actual", "Count"], 29)


if __name__ == "__main__":
    unittest.main()
