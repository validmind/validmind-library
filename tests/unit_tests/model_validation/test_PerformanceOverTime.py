import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.PerformanceOverTime import PerformanceOverTime


def _dataset(break_size, n_per_period=300, seed=0, missing_periods=0):
    rng = np.random.default_rng(seed)
    periods = [f"2024-{m:02d}" for m in range(1, 13)]
    df = pd.DataFrame({"period": np.repeat(periods, n_per_period)})
    df["x"] = rng.normal(0, 1, len(df))
    prediction = 2 * df.x
    after_break = df.period >= "2024-10"
    df["y"] = prediction + rng.normal(0, 1, len(df)) + break_size * after_break
    df = df.sample(frac=1, random_state=0)  # periods arrive unsorted
    df.loc[df.index[:missing_periods], "period"] = None
    model = vm.init_model(input_id="m", predict_fn=lambda row: 0.0, __log=False)
    dataset = vm.init_dataset(
        df, input_id="d", target_column="y", feature_columns=["x"],
        extra_columns={"period": None}, __log=False,
    )
    dataset.assign_predictions(model, prediction_values=(2 * df.x).to_numpy())
    return dataset, model


class TestPerformanceOverTime(unittest.TestCase):
    def test_flags_periods_after_a_break(self):
        table, fig, passed, _ = PerformanceOverTime(*_dataset(break_size=0.5), time_column="period")

        self.assertFalse(passed)
        self.assertEqual(list(table["Period"]), sorted(table["Period"]))
        self.assertEqual(list(table[table.Result == "Biased"]["Period"]), ["2024-10", "2024-11", "2024-12"])
        self.assertIsInstance(fig, go.Figure)

    def test_stable_model_passes(self):
        _, _, passed, _ = PerformanceOverTime(*_dataset(break_size=0.0), time_column="period")
        self.assertTrue(passed)

    def test_rows_without_a_period_are_reported_last(self):
        table = PerformanceOverTime(
            *_dataset(break_size=0.0, missing_periods=20), time_column="period"
        )[0]

        self.assertEqual(table["Period"].iloc[-1], "Missing")
        self.assertEqual(table["Count"].iloc[-1], 20)
        self.assertEqual(table["Count"].sum(), 3600)

    def test_time_column_must_exist(self):
        with self.assertRaisesRegex(ValueError, "not in the dataset"):
            PerformanceOverTime(*_dataset(break_size=0.0), time_column="missing")


if __name__ == "__main__":
    unittest.main()
