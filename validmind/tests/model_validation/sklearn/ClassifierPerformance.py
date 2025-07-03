# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    return roc_auc_score(lb.transform(y_test), lb.transform(y_pred), average=average)


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def ClassifierPerformance(
    dataset: VMDataset, model: VMModel, average: str = "macro"
) -> Dict[str, List[Dict[str, float]]]:
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
    y_pred = dataset.y_pred(model)
    y_true = dataset.y

    labels = np.unique(y_true)
    labels = sorted(labels.tolist())

    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0,
    )

    if len(labels) > 2:
        y_true = y_true.astype(y_pred.dtype)
        roc_auc = multiclass_roc_auc_score(y_true, y_pred, average=average)
    else:
        y_prob = dataset.y_prob(model)
        y_true = y_true.astype(y_prob.dtype).flatten()
        roc_auc = roc_auc_score(y_true, y_prob, average=average)

    report["roc_auc"] = roc_auc

    pr_f1_table = [
        {
            "Class": f"{class_name}",
            "Precision": report[f"{class_name}"]["precision"],
            "Recall": report[f"{class_name}"]["recall"],
            "F1": report[f"{class_name}"]["f1-score"],
        }
        for class_name in labels
    ]

    for avg in ["weighted avg", "macro avg"]:
        pr_f1_table.append(
            {
                "Class": avg.replace("avg", "Average").title(),
                "Precision": report[avg]["precision"],
                "Recall": report[avg]["recall"],
                "F1": report[avg]["f1-score"],
            }
        )

    return {
        "Precision, Recall, and F1": pr_f1_table,
        "Accuracy and ROC AUC": [
            {"Metric": m, "Value": report[k]}
            for m, k in [("Accuracy", "accuracy"), ("ROC AUC", "roc_auc")]
        ],
    }
