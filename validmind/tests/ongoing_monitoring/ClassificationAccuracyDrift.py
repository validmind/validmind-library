# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from typing import List
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def ClassificationAccuracyDrift(
    datasets: List[VMDataset], model: VMModel, drift_pct_threshold=20
):
    """
    Compares classification accuracy metrics between reference and monitoring datasets.

    ### Purpose
    This test evaluates drift in classification accuracy metrics including per-label and
    macro-averaged precision, recall, and F1 scores.

    ### Test Mechanism
    Calculates classification metrics for both reference and monitoring datasets and
    compares them to identify significant changes in model performance.

    ### Signs of High Risk
    - Large drifts in accuracy metrics (above threshold)
    - Inconsistent changes across different labels
    - Significant drops in macro-averaged metrics
    """
    # Get predictions and true values
    y_true_ref = datasets[0].y
    y_pred_ref = datasets[0].y_pred(model)

    y_true_mon = datasets[1].y
    y_pred_mon = datasets[1].y_pred(model)

    # Get unique labels from reference dataset
    labels = np.unique(y_true_ref)
    labels = sorted(labels.tolist())

    # Calculate classification reports
    report_ref = classification_report(
        y_true=y_true_ref,
        y_pred=y_pred_ref,
        output_dict=True,
        zero_division=0,
    )

    report_mon = classification_report(
        y_true=y_true_mon,
        y_pred=y_pred_mon,
        output_dict=True,
        zero_division=0,
    )

    # Create metrics dataframe
    metrics = []

    # Add accuracy
    metrics.append(
        {
            "Metric": "Accuracy",
            "Reference": report_ref["accuracy"],
            "Monitoring": report_mon["accuracy"],
        }
    )

    # Add per-label metrics
    for label in labels:
        label_str = str(label)
        for metric in ["precision", "recall", "f1-score"]:
            metric_name = f"{metric.title()}_{label_str}"
            metrics.append(
                {
                    "Metric": metric_name,
                    "Reference": report_ref[label_str][metric],
                    "Monitoring": report_mon[label_str][metric],
                }
            )

    # Add macro averages
    for metric in ["precision", "recall", "f1-score"]:
        metric_name = f"{metric.title()}_Macro"
        metrics.append(
            {
                "Metric": metric_name,
                "Reference": report_ref["macro avg"][metric],
                "Monitoring": report_mon["macro avg"][metric],
            }
        )

    # Create DataFrame
    df = pd.DataFrame(metrics)

    # Calculate drift percentage with direction
    df["Drift (%)"] = (
        (df["Monitoring"] - df["Reference"]) / df["Reference"].abs() * 100
    ).round(2)

    # Add Pass/Fail column based on absolute drift
    df["Pass/Fail"] = (
        df["Drift (%)"]
        .abs()
        .apply(lambda x: "Pass" if x < drift_pct_threshold else "Fail")
    )

    # Calculate overall pass/fail
    pass_fail_bool = (df["Pass/Fail"] == "Pass").all()

    return ({"Classification Accuracy Metrics": df}, pass_fail_bool)
