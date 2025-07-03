# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

from .ClassifierPerformance import multiclass_roc_auc_score


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "model_performance",
    "model_comparison",
)
@tasks("classification", "text_classification")
def ModelsPerformanceComparison(
    dataset: VMDataset, models: list[VMModel]
) -> Dict[str, List[Dict[str, float]]]:
    """
    Evaluates and compares the performance of multiple Machine Learning models using various metrics like accuracy,
    precision, recall, and F1 score.

    ### Purpose

    The Models Performance Comparison test aims to evaluate and compare the performance of various Machine Learning
    models using test data. It employs multiple metrics such as accuracy, precision, recall, and the F1 score, among
    others, to assess model performance and assist in selecting the most effective model for the designated task.

    ### Test Mechanism

    The test employs Scikit-learn’s performance metrics to evaluate each model's performance for both binary and
    multiclass classification tasks. To compare performances, the test runs each model against the test dataset, then
    produces a comprehensive classification report. This report includes metrics such as accuracy, precision, recall,
    and the F1 score. Based on whether the task at hand is binary or multiclass classification, it calculates metrics
    for all the classes and their weighted averages, macro averages, and per-class metrics. The test will be skipped if
    no models are supplied.

    ### Signs of High Risk

    - Low scores in accuracy, precision, recall, and F1 metrics indicate a potentially high risk.
    - A low area under the Receiver Operating Characteristic (ROC) curve (roc_auc score) is another possible indicator
    of high risk.
    - If the metrics scores are significantly lower than alternative models, this might suggest a high risk of failure.

    ### Strengths

    - Provides a simple way to compare the performance of multiple models, accommodating both binary and multiclass
    classification tasks.
    - Offers a holistic view of model performance through a comprehensive report of key performance metrics.
    - The inclusion of the ROC AUC score is advantageous, as this robust performance metric can effectively handle
    class imbalance issues.

    ### Limitations

    - May not be suitable for more complex performance evaluations that consider factors such as prediction speed,
    computational cost, or business-specific constraints.
    - The test's reliability depends on the provided test dataset; hence, the selected models' performance could vary
    with unseen data or changes in the data distribution.
    - The ROC AUC score might not be as meaningful or easily interpretable for multilabel/multiclass tasks.
    """
    y_true = dataset.y
    classes = {str(i) for i in np.unique(y_true)}

    prf_table = []
    acc_roc_auc_table = []

    for model in models:
        y_pred = dataset.y_pred(model)

        report = classification_report(y_true, y_pred, output_dict=True)
        report["roc_auc"] = multiclass_roc_auc_score(y_true, y_pred)

        for class_name in classes:
            prf_table.append(
                {
                    "Model": model.input_id,
                    "Class": class_name,
                    "Precision": report[class_name]["precision"],
                    "Recall": report[class_name]["recall"],
                    "F1-Score": report[class_name]["f1-score"],
                }
            )
        for avg_metric in ["weighted avg", "macro avg"]:
            prf_table.append(
                {
                    "Model": model.input_id,
                    "Class": avg_metric,
                    "Precision": report[avg_metric]["precision"],
                    "Recall": report[avg_metric]["recall"],
                    "F1-Score": report[avg_metric]["f1-score"],
                }
            )

        for metric in ["accuracy", "roc_auc"]:
            acc_roc_auc_table.append(
                {
                    "Model": model.input_id,
                    "Metric": metric,
                    "Value": report[metric],
                }
            )

    return {
        "Precision, Recall, and F1 Comparison": prf_table,
        "Accuracy and ROC AUC Comparison": acc_roc_auc_table,
    }
