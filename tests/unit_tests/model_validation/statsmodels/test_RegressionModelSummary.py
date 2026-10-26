import unittest

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import validmind as vm
from validmind.tests.model_validation.statsmodels.RegressionModelSummary import (
    RegressionModelSummary,
)


class TestRegressionModelSummary(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        df["y"] = 2 * df.x + rng.normal(0, 1, 200)
        fit = smf.ols("y ~ x", df).fit()

        self.model = vm.init_model(fit, input_id="m", __log=False)
        self.dataset = vm.init_dataset(df, input_id="d", target_column="y", __log=False)
        self.dataset.assign_predictions(self.model, prediction_values=fit.predict(df).values)

    def test_rmse_is_the_root_of_mse(self):
        # `mean_squared_error(squared=False)` was removed in scikit-learn 1.6
        rows, _ = RegressionModelSummary(self.dataset, self.model)

        self.assertAlmostEqual(rows[0]["RMSE"], np.sqrt(rows[0]["MSE"]))
        self.assertGreater(rows[0]["R-Squared"], 0.7)


if __name__ == "__main__":
    unittest.main()
