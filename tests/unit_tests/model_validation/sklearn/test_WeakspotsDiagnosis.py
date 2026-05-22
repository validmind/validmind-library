import unittest

from validmind.tests.model_validation.sklearn.WeakspotsDiagnosis import (
    _prepare_metrics_and_thresholds,
)


class TestWeakspotsDiagnosisThresholds(unittest.TestCase):
    def test_partial_thresholds_use_defaults_for_plotting(self):
        _, plot_thresholds, pass_thresholds = _prepare_metrics_and_thresholds(
            metrics=None,
            thresholds={"accuracy": 0.65},
        )

        self.assertEqual(pass_thresholds, {"Accuracy": 0.65})
        self.assertEqual(plot_thresholds["Accuracy"], 0.65)
        self.assertEqual(plot_thresholds["Precision"], 0.5)
        self.assertEqual(plot_thresholds["Recall"], 0.5)
        self.assertEqual(plot_thresholds["F1"], 0.7)

    def test_partial_thresholds_subset_for_pass_fail(self):
        _, _, pass_thresholds = _prepare_metrics_and_thresholds(
            metrics=None,
            thresholds={"accuracy": 0.75, "f1": 0.55},
        )

        self.assertEqual(set(pass_thresholds.keys()), {"Accuracy", "F1"})


if __name__ == "__main__":
    unittest.main()
