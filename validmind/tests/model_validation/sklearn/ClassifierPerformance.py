# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from validmind.vm_models import Metric, ResultSummary, ResultTable, ResultTableMetadata


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)

    return roc_auc_score(lb.transform(y_test), lb.transform(y_pred), average=average)


@dataclass
class ClassifierPerformance(Metric):
    """
    Evaluates performance of binary or multiclass classification models using precision, recall, F1-Score, accuracy,
    and ROC AUC scores.

    ### Purpose

    The Classifier Performance test is designed to evaluate the performance of Machine Learning classification models.
    It accomplishes this by computing precision, recall, F1-Score, and accuracy, as well as the ROC AUC (Receiver
    operating characteristic - Area under the curve) scores, thereby providing a comprehensive analytic view of the
    models' performance. The test is adaptable, handling binary and multiclass models equally effectively.

    ### Test Mechanism

    The test produces a report that includes precision, recall, F1-Score, and accuracy, by leveraging the
    `classification_report` from scikit-learn's metrics module. For multiclass models, macro and weighted averages for
    these scores are also calculated. Additionally, the ROC AUC scores are calculated and included in the report using
    the `multiclass_roc_auc_score` function. The outcome of the test (report format) differs based on whether the model
    is binary or multiclass.

    ### Signs of High Risk

    - Low values for precision, recall, F1-Score, accuracy, and ROC AUC, indicating poor performance.
    - Imbalance in precision and recall scores.
    - A low ROC AUC score, especially scores close to 0.5 or lower, suggesting a failing model.

    ### Strengths

    - Versatile, capable of assessing both binary and multiclass models.
    - Utilizes a variety of commonly employed performance metrics, offering a comprehensive view of model performance.
    - The use of ROC-AUC as a metric is beneficial for evaluating unbalanced datasets.

    ### Limitations

    - Assumes correctly identified labels for binary classification models.
    - Specifically designed for classification models and not suitable for regression models.
    - May provide limited insights if the test dataset does not represent real-world scenarios adequately.
    """

    name = "classifier_performance"
    required_inputs = ["model", "dataset"]
    tasks = ["classification", "text_classification"]
    tags = [
        "sklearn",
        "binary_classification",
        "multiclass_classification",
        "model_performance",
    ]
    default_params = {"average": "macro"}

    def summary(self, metric_value: dict):
        """
        When building a multi-class summary we need to calculate weighted average,
        macro average and per class metrics.
        """
        classes = {str(i) for i in np.unique(self.inputs.dataset.y)}
        pr_f1_table = [
            {
                "Class": class_name,
                "Precision": metric_value[class_name]["precision"],
                "Recall": metric_value[class_name]["recall"],
                "F1": metric_value[class_name]["f1-score"],
            }
            for class_name in classes
        ]
        pr_f1_table.extend(
            [
                {
                    "Class": "Weighted Average",
                    "Precision": metric_value["weighted avg"]["precision"],
                    "Recall": metric_value["weighted avg"]["recall"],
                    "F1": metric_value["weighted avg"]["f1-score"],
                },
                {
                    "Class": "Macro Average",
                    "Precision": metric_value["macro avg"]["precision"],
                    "Recall": metric_value["macro avg"]["recall"],
                    "F1": metric_value["macro avg"]["f1-score"],
                },
            ]
        )

        acc_roc_auc_table = [
            {
                "Metric": "Accuracy" if metric_name == "accuracy" else "ROC AUC",
                "Value": metric_value[metric_name],
            }
            for metric_name in ["accuracy", "roc_auc"]
        ]

        return ResultSummary(
            results=[
                ResultTable(
                    data=pr_f1_table,
                    metadata=ResultTableMetadata(title="Precision, Recall, and F1"),
                ),
                ResultTable(
                    data=acc_roc_auc_table,
                    metadata=ResultTableMetadata(title="Accuracy and ROC AUC"),
                ),
            ]
        )

    def run(self):
        report = classification_report(
            self.inputs.dataset.y,
            self.inputs.dataset.y_pred(self.inputs.model),
            output_dict=True,
            zero_division=0,
        )

        y_true = self.inputs.dataset.y

        if len(np.unique(y_true)) > 2:
            y_pred = self.inputs.dataset.y_pred(self.inputs.model)
            y_true = y_true.astype(y_pred.dtype)
            roc_auc = multiclass_roc_auc_score(
                y_true, y_pred, average=self.params["average"]
            )
        else:
            y_prob = self.inputs.dataset.y_prob(self.inputs.model)
            y_true = y_true.astype(y_prob.dtype).flatten()
            roc_auc = roc_auc_score(y_true, y_prob, average=self.params["average"])

        report["roc_auc"] = roc_auc

        return self.cache_results(report)
