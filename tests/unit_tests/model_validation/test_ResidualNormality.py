import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind.tests.model_validation.ResidualNormality import ResidualNormality


def _setup(noise, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    df["y"] = 2 * df.x + noise(rng, n)
    model = vm.init_model(
        LinearRegression().fit(df[["x"]], df.y), input_id="m", __log=False
    )
    dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
    dataset.assign_predictions(model)
    return dataset, model


class TestResidualNormality(unittest.TestCase):
    def test_passes_normal_errors(self):
        dataset, model = _setup(lambda rng, n: rng.normal(0, 1, n))
        tables, fig, passed, _ = ResidualNormality(dataset, model)

        self.assertTrue(passed)
        self.assertTrue((tables["Normality Tests"]["Pass/Fail"] == "Pass").all())
        self.assertIsInstance(fig, go.Figure)

    def test_fails_skewed_errors(self):
        dataset, model = _setup(lambda rng, n: rng.exponential(1, n))
        tables, _, passed, _ = ResidualNormality(dataset, model)

        self.assertFalse(passed)
        self.assertTrue((tables["Normality Tests"]["Pass/Fail"] == "Fail").all())
        self.assertGreater(tables["Residual Shape"]["Skewness"].iloc[0], 1)

    def test_tests_residuals_not_features(self):
        # skewed feature, normal errors: feature normality is not an assumption
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.exponential(1, 1000)})
        df["y"] = 2 * df.x + rng.normal(0, 1, 1000)
        model = vm.init_model(
            LinearRegression().fit(df[["x"]], df.y), input_id="m", __log=False
        )
        dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
        dataset.assign_predictions(model)

        self.assertTrue(ResidualNormality(dataset, model)[2])

    def test_samples_large_datasets_for_shapiro(self):
        dataset, model = _setup(lambda rng, n: rng.normal(0, 1, n), n=6000)
        tests = ResidualNormality(dataset, model)[0]["Normality Tests"]
        self.assertEqual(tests["Test"].iloc[1], "Shapiro-Wilk (n=5000)")


if __name__ == "__main__":
    unittest.main()
