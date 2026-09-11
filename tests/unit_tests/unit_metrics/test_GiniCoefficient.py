import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import validmind as vm
from validmind.unit_metrics.regression.GiniCoefficient import GiniCoefficient


def _build(noise):
    """Fit a linear model on data whose residual scale is set by ``noise``."""
    rng = np.random.RandomState(42)
    X = np.arange(200).reshape(-1, 1)
    y = 2 * X.ravel() + rng.normal(0, noise, 200)

    df = pd.DataFrame({"feature": X.ravel(), "target": y})
    model = LinearRegression().fit(X, y)

    vm_model = vm.init_model(input_id=f"gini_model_{noise}", model=model, __log=False)
    vm_dataset = vm.init_dataset(
        input_id=f"gini_dataset_{noise}",
        dataset=df,
        target_column="target",
        __log=False,
    )
    vm_dataset.assign_predictions(vm_model)

    return vm_dataset, vm_model


class TestGiniCoefficient(unittest.TestCase):
    def test_returns_finite_float(self):
        """Covers the trapezoid call, which had no test at all.

        `np.trapz` was removed in NumPy 2, so this raised AttributeError on any
        Python 3.14 environment (numpy>=2.3) before the fallback was added.
        """
        dataset, model = _build(noise=5)
        gini = GiniCoefficient(dataset, model)

        self.assertIsInstance(gini, float)
        self.assertTrue(np.isfinite(gini))

    def test_tighter_fit_scores_nearer_zero(self):
        """Predictions tracking the target put the Lorenz curve on the diagonal.

        A near-perfect fit gives an area of ~0.5 and so a coefficient of ~0; the
        noisier the residuals, the further the coefficient moves away from it.
        """
        tight_dataset, tight_model = _build(noise=1)
        loose_dataset, loose_model = _build(noise=400)

        tight = GiniCoefficient(tight_dataset, tight_model)
        loose = GiniCoefficient(loose_dataset, loose_model)

        self.assertAlmostEqual(tight, 0.0, places=3)
        self.assertGreater(abs(loose), abs(tight))


if __name__ == "__main__":
    unittest.main()
