import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import validmind as vm
from validmind.tests.model_validation.ResidualBiasBySegment import ResidualBiasBySegment


def _dataset(n=4000, seed=0, bias=0.0, missing_scores=0):
    """`bias` is added to y for group == 'hit'; predictions are unbiased overall, as a
    fitted model's would be (its intercept absorbs the average bias)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "group": rng.choice(["hit", "a", "b", "c"], n),
            "score": rng.uniform(300, 850, n),
            "flag": rng.integers(0, 2, n),
        }
    )
    df["y"] = 2 * df.x + rng.normal(0, 1, n) + bias * (df.group == "hit")
    df.loc[df.index[:missing_scores], "score"] = np.nan
    prediction = 2 * df.x + bias * (df.group == "hit").mean()
    model = vm.init_model(input_id="m", predict_fn=lambda row: 0.0, __log=False)
    dataset = vm.init_dataset(
        df,
        input_id="d",
        target_column="y",
        feature_columns=["x"],
        extra_columns={"group": None, "score": None, "flag": None},
        __log=False,
    )
    dataset.assign_predictions(model, prediction_values=prediction.to_numpy())
    return dataset, model


class TestResidualBiasBySegment(unittest.TestCase):
    def test_flags_only_the_column_carrying_the_bias(self):
        dataset, model = _dataset(bias=1.0)
        table, fig, passed, _ = ResidualBiasBySegment(dataset, model)
        biased = table[table.Result == "Biased"].set_index("Segment")

        self.assertFalse(passed)
        self.assertEqual(set(biased["Segment Column"]), {"group"})
        # 'hit' is under-predicted; the rest are over-predicted, averaging to zero
        self.assertAlmostEqual(biased.loc["hit", "Mean Residual"], 0.75, delta=0.15)
        self.assertTrue((biased.drop("hit")["Mean Residual"] < 0).all())
        self.assertIsInstance(fig, go.Figure)

    def test_quiet_without_bias_across_many_segments(self):
        dataset, model = _dataset(bias=0.0)
        table, _, passed, _ = ResidualBiasBySegment(dataset, model, n_bins=10)
        self.assertTrue(passed)
        self.assertGreater(len(table), 15)

    def test_default_segments_include_columns_the_model_does_not_use(self):
        dataset, model = _dataset()
        columns = set(ResidualBiasBySegment(dataset, model)[0]["Segment Column"])
        self.assertEqual(columns, {"x", "group", "score", "flag"})

    def test_numeric_columns_are_binned_from_their_real_minimum(self):
        dataset, model = _dataset()
        table = ResidualBiasBySegment(
            dataset, model, segment_columns=["score"], n_bins=4
        )[0]
        self.assertEqual(len(table), 4)
        first_edge = float(table["Segment"].iloc[0].split(" – ")[0].replace(",", ""))
        self.assertAlmostEqual(first_edge, dataset.df.score.min(), delta=1)

    def test_missing_values_are_their_own_segment_listed_last(self):
        dataset, model = _dataset(missing_scores=10)
        table = ResidualBiasBySegment(
            dataset, model, segment_columns=["score"], n_bins=4
        )[0]

        self.assertEqual(table["Segment"].iloc[-1], "Missing")
        self.assertEqual(table["Count"].iloc[-1], 10)

    def test_column_pairs_segment_on_combinations(self):
        dataset, model = _dataset()
        table = ResidualBiasBySegment(
            dataset, model, segment_columns=[["group", "flag"]]
        )[0]
        self.assertEqual(set(table["Segment Column"]), {"group × flag"})
        self.assertEqual(len(table), 8)

    def test_small_segments_are_reported_not_assessed(self):
        dataset, model = _dataset(n=200, bias=5.0)
        table = ResidualBiasBySegment(
            dataset, model, segment_columns=["group"], min_segment_size=1000
        )[0]
        self.assertTrue((table.Result == "Too small to assess").all())

    def test_unknown_column_explains_extra_columns(self):
        dataset, model = _dataset()
        with self.assertRaisesRegex(ValueError, "add_extra_column"):
            ResidualBiasBySegment(dataset, model, segment_columns=["missing"])

    def test_large_values_keep_one_segment_per_bin(self):
        n = 1000
        df = pd.DataFrame({"x": np.zeros(n), "date": 20240101 + np.arange(n) % 365})
        df["y"] = np.random.default_rng(0).normal(0, 1, n)
        model = vm.init_model(input_id="m", predict_fn=lambda row: 0.0, __log=False)
        dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
        dataset.assign_predictions(model, prediction_values=np.zeros(n))
        table = ResidualBiasBySegment(
            dataset, model, segment_columns=["date"], n_bins=4
        )[0]

        self.assertEqual(len(table), 4)
        self.assertEqual(table["Segment"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
