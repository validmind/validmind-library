import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind.tests.model_validation.FeatureSensitivity import FeatureSensitivity


def _data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "big": rng.normal(10, 2, n),
            "small": rng.normal(0, 1, n),
            "flag": (rng.random(n) < 0.25).astype(int),
        }
    )
    df["y"] = 3 * df.big + 0.5 * df.small + 4 * df.flag + rng.normal(0, 0.1, n)
    return df


def _sklearn(df):
    fit = LinearRegression().fit(df[["big", "small", "flag"]], df.y)
    model = vm.init_model(fit, input_id="m", __log=False)
    dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
    dataset.assign_predictions(model)
    return dataset, model


class TestFeatureSensitivity(unittest.TestCase):
    def setUp(self):
        self.df = _data()

    def test_linear_effects_in_target_units(self):
        dataset, model = _sklearn(self.df)
        table, fig, _ = FeatureSensitivity(dataset, model, clip_to_range=False)
        rows = table.set_index("Feature")
        std = self.df.big.std()

        self.assertAlmostEqual(rows.loc["big", "Change (Up)"], 3 * std, delta=0.05)
        self.assertAlmostEqual(rows.loc["big", "Change (Down)"], -3 * std, delta=0.05)
        self.assertEqual(table["Feature"].iloc[0], "big")  # ranked by effect
        self.assertIsInstance(fig, go.Figure)

    def test_binary_features_are_flipped(self):
        dataset, model = _sklearn(self.df)
        rows = FeatureSensitivity(dataset, model)[0].set_index("Feature")
        share = self.df.flag.mean()

        self.assertEqual(rows.loc["flag", "Shock"], "0 → 1")
        self.assertAlmostEqual(rows.loc["flag", "Change (Up)"], 4 * (1 - share), delta=0.05)
        self.assertAlmostEqual(rows.loc["flag", "Change (Down)"], -4 * share, delta=0.05)

    def test_custom_shock_in_feature_units(self):
        dataset, model = _sklearn(self.df)
        rows = FeatureSensitivity(
            dataset, model, features=["small"], shocks={"small": 0.1}, clip_to_range=False
        )[0].set_index("Feature")

        self.assertEqual(list(rows.index), ["small"])
        self.assertAlmostEqual(rows.loc["small", "Change (Up)"], 0.05, delta=0.005)

    def test_shocks_are_clipped_to_the_observed_range(self):
        dataset, model = _sklearn(self.df)
        huge = {"big": 100.0}
        clipped = FeatureSensitivity(dataset, model, features=["big"], shocks=huge)[0]
        unclipped = FeatureSensitivity(
            dataset, model, features=["big"], shocks=huge, clip_to_range=False
        )[0]
        max_up = 3 * (self.df.big.max() - self.df.big.mean())

        self.assertAlmostEqual(clipped["Change (Up)"].iloc[0], max_up, delta=0.1)
        self.assertGreater(unclipped["Change (Up)"].iloc[0], 299)

    def test_statsmodels_formula_and_array_apis(self):
        formula = smf.ols("y ~ big + small + flag", self.df).fit()
        array = sm.OLS(self.df.y, sm.add_constant(self.df[["big", "small", "flag"]])).fit()
        tables = []
        for fit, name in [(formula, "f"), (array, "a")]:
            model = vm.init_model(fit, input_id=name, __log=False)
            dataset = vm.init_dataset(
                self.df.copy(),
                input_id="d",
                target_column="y",
                feature_columns=["big", "small", "flag"],
                __log=False,
            )
            dataset.assign_predictions(model, prediction_values=np.zeros(len(self.df)))
            tables.append(FeatureSensitivity(dataset, model)[0])

        pd.testing.assert_frame_equal(tables[0], tables[1])

    def test_non_linear_model_is_asymmetric(self):
        df = self.df.assign(y=np.exp(self.df.big / 4))
        model = vm.init_model(
            input_id="exp", predict_fn=lambda row: np.exp(row["big"] / 4), __log=False
        )
        dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
        dataset.assign_predictions(model, prediction_values=df.y.to_numpy())
        row = FeatureSensitivity(dataset, model, features=["big"], clip_to_range=False)[0]

        self.assertGreater(row["Change (Up)"].iloc[0], -row["Change (Down)"].iloc[0])


if __name__ == "__main__":
    unittest.main()
