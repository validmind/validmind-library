import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import validmind as vm
from validmind import RawData
from validmind.tests.ongoing_monitoring.TargetPredictionDistributionPlot import (
    TargetPredictionDistributionPlot,
)


class TestTargetPredictionDistributionPlot(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(
            n_samples=600,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=2,
            random_state=7,
        )
        model = LogisticRegression(max_iter=1000).fit(X, y)
        self.vm_model = vm.init_model(input_id="tpd_model", model=model, __log=False)

        def _dataset(input_id, rows):
            df = pd.DataFrame(X[rows], columns=[f"f{i}" for i in range(5)])
            df["target"] = y[rows]
            ds = vm.init_dataset(
                input_id=input_id, dataset=df, target_column="target", __log=False
            )
            ds.assign_predictions(self.vm_model)

            return ds

        self.datasets = [
            _dataset("reference", slice(0, 300)),
            _dataset("monitoring", slice(300, 600)),
        ]

    def test_return_structure(self):
        tables, fig, passed, raw_data = TargetPredictionDistributionPlot(
            self.datasets, self.vm_model
        )

        self.assertIn("Distribution Moments", tables)
        moments = tables["Distribution Moments"]
        self.assertIn("Drift (%)", moments.columns)
        self.assertIn("Pass/Fail", moments.columns)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(passed, (bool, np.bool_))
        self.assertIsInstance(raw_data, RawData)

    def test_two_kde_curves(self):
        """One filled KDE curve per dataset, each spanning its own value range."""
        _, fig, _, _ = TargetPredictionDistributionPlot(self.datasets, self.vm_model)

        self.assertEqual(len(fig.data), 2)
        self.assertEqual(
            [t.name for t in fig.data],
            ["Reference Prediction", "Monitor Prediction"],
        )

        for trace, dataset in zip(fig.data, self.datasets):
            self.assertEqual(trace.type, "scatter")
            self.assertEqual(len(trace.x), 500)

            probabilities = dataset.y_prob_df(self.vm_model).iloc[:, 0].values
            self.assertAlmostEqual(trace.x[0], probabilities.min())
            self.assertLessEqual(trace.x[-1], probabilities.max())
            self.assertTrue(np.all(np.asarray(trace.y) > 0))


if __name__ == "__main__":
    unittest.main()
