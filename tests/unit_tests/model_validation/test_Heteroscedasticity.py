import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind.errors import SkipTestError
from validmind.tests.model_validation.Heteroscedasticity import Heteroscedasticity


def _setup(heteroscedastic, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"x": rng.uniform(1, 10, n), "z": rng.normal(0, 1, n), "group": rng.choice(["a", "b"], n)}
    )
    scale = df.x if heteroscedastic else 3.0
    df["y"] = 2 * df.x + df.z + rng.normal(0, 1, n) * scale
    return df


def _sklearn(df):
    fit = LinearRegression().fit(df[["x", "z"]], df.y)
    model = vm.init_model(fit, input_id="m", __log=False)
    dataset = vm.init_dataset(
        df, input_id="d", target_column="y", feature_columns=["x", "z", "group"], __log=False
    )
    # the model uses x and z only; "group" is a categorical feature of the dataset
    dataset.assign_predictions(model, prediction_values=fit.predict(df[["x", "z"]]))
    return dataset, model


class TestHeteroscedasticity(unittest.TestCase):
    def test_detects_variance_growing_with_a_feature(self):
        tables, fig, passed, _ = Heteroscedasticity(*_sklearn(_setup(True)))
        spread = tables["Residual Spread by Prediction Bin"]["Residual Std. Deviation"]

        self.assertFalse(passed)
        self.assertTrue((tables["Heteroscedasticity Tests"]["Pass/Fail"] == "Fail").all())
        self.assertGreater(spread.iloc[-1], 2 * spread.iloc[0])
        self.assertIsInstance(fig, go.Figure)

    def test_passes_constant_variance(self):
        tables, _, passed, _ = Heteroscedasticity(*_sklearn(_setup(False)))
        self.assertTrue(passed)

    def test_works_with_statsmodels_predictions(self):
        df = _setup(True)
        fit = smf.ols("y ~ x + z", df).fit()
        model = vm.init_model(fit, input_id="sm", __log=False)
        dataset = vm.init_dataset(
            df, input_id="d", target_column="y", feature_columns=["x", "z", "group"], __log=False
        )
        dataset.assign_predictions(model, prediction_values=fit.predict(df).values)

        self.assertFalse(Heteroscedasticity(dataset, model)[2])

    def test_skips_without_numeric_features(self):
        df = _setup(False)
        model = vm.init_model(input_id="f", predict_fn=lambda row: 0.0, __log=False)
        dataset = vm.init_dataset(
            df[["group", "y"]], input_id="d", target_column="y", __log=False
        )
        dataset.assign_predictions(model, prediction_values=np.zeros(len(df)))

        with self.assertRaises(SkipTestError):
            Heteroscedasticity(dataset, model)


if __name__ == "__main__":
    unittest.main()
