from dataclasses import dataclass
from typing import List
import numpy as np

import pandas as pd
from sklearn import metrics

from validmind.vm_models import (
    ResultSummary,
    ResultTable,
    ResultTableMetadata,
    TestResult,
    ThresholdTest,
)


@dataclass
class MinimumROCAUCScore(ThresholdTest):
    """
    Test that the model's ROC AUC score on the validation dataset meets or
    exceeds a predefined threshold.
    """

    category = "model_performance"
    name = "roc_auc_score"
    required_context = ["model"]
    default_params = {"min_threshold": 0.5}

    def summary(self, results: List[TestResult], all_passed: bool):
        """
        The roc auc score test returns results like these:
        [{"values": {"score": 0.734375, "threshold": 0.7}, "passed": true}]
        """
        result = results[0]
        results_table = [
            {
                "Score": result.values["score"],
                "Threshold": result.values["threshold"],
                "Pass/Fail": "Pass" if result.passed else "Fail",
            }
        ]

        return ResultSummary(
            results=[
                ResultTable(
                    data=pd.DataFrame(results_table),
                    metadata=ResultTableMetadata(title="Minimum ROC AUC Score Test"),
                )
            ]
        )

    def run(self):
        if self.model.device_type and self.model._is_pytorch_model:
            if not self.model.device_type == "gpu":
                y_true = np.array(self.model.test_ds.y.cpu())
            else:
                y_true = np.array(self.model.test_ds.y)
        else:
            y_true = self.model.test_ds.y

        class_pred = self.model.class_predictions(self.model.y_test_predict)
        roc_auc = metrics.roc_auc_score(y_true, class_pred)

        passed = roc_auc > self.params["min_threshold"]
        results = [
            TestResult(
                passed=passed,
                values={
                    "score": roc_auc,
                    "threshold": self.params["min_threshold"],
                },
            )
        ]

        return self.cache_results(results, passed=all([r.passed for r in results]))