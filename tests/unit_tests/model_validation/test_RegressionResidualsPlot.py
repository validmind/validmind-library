import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind import RawData
from validmind.tests.model_validation.RegressionResidualsPlot import (
    RegressionResidualsPlot,
)


class TestRegressionResidualsPlot(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X = np.arange(200).reshape(-1, 1)
        y = 2 * X.ravel() + np.random.normal(0, 5, 200)

        df = pd.DataFrame({"feature": X.ravel(), "target": y})
        model = LinearRegression().fit(X, y)

        self.vm_model = vm.init_model(input_id="rrp_model", model=model, __log=False)
        self.vm_dataset = vm.init_dataset(
            input_id="rrp_dataset", dataset=df, target_column="target", __log=False
        )
        self.vm_dataset.assign_predictions(self.vm_model)

    def test_return_structure(self):
        result = RegressionResidualsPlot(self.vm_model, self.vm_dataset, bin_size=1.0)

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], go.Figure)
        self.assertIsInstance(result[1], go.Figure)
        self.assertIsInstance(result[2], RawData)

    def test_residuals_figure_has_histogram_and_kde(self):
        """The residuals figure is a density histogram overlaid with a KDE curve.

        Both must use the same normalisation, or the curve sits orders of
        magnitude away from the bars and the plot is silently useless.
        """
        fig, _, _ = RegressionResidualsPlot(
            self.vm_model, self.vm_dataset, bin_size=1.0
        )

        hist = next(t for t in fig.data if t.type == "histogram")
        curve = next(t for t in fig.data if t.type == "scatter")

        self.assertEqual(hist.histnorm, "probability density")
        self.assertEqual(hist.xbins.size, 1.0)

        residuals = (
            self.vm_dataset.y.flatten()
            - self.vm_dataset.y_pred(self.vm_model).flatten()
        )
        self.assertAlmostEqual(curve.x[0], residuals.min())
        self.assertLessEqual(curve.x[-1], residuals.max())
        self.assertEqual(len(curve.x), 500)
        self.assertTrue(np.all(np.asarray(curve.y) > 0))

        # A density curve integrates to ~1 over the data range.
        self.assertAlmostEqual(np.trapz(curve.y, curve.x), 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
