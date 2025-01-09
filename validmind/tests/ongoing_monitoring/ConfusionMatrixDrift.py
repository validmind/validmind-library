# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def ConfusionMatrixDrift(
    datasets: List[VMDataset], model: VMModel, drift_pct_threshold=20
):
    """
    Compares confusion matrix metrics between reference and monitoring datasets.

    ### Purpose
    This test evaluates drift in confusion matrix elements including True Positives,
    True Negatives, False Positives, and False Negatives.

    ### Test Mechanism
    Calculates confusion matrices for both reference and monitoring datasets and
    compares corresponding elements to identify significant changes in model predictions.

    ### Signs of High Risk
    - Large drifts in confusion matrix elements (above threshold)
    - Significant changes in error patterns (FP, FN)
    - Inconsistent changes across different classes
    """
    # Get predictions and true values for reference dataset
    y_pred_ref = datasets[0].y_pred(model)
    y_true_ref = datasets[0].y.astype(y_pred_ref.dtype)

    # Get predictions and true values for monitoring dataset
    y_pred_mon = datasets[1].y_pred(model)
    y_true_mon = datasets[1].y.astype(y_pred_mon.dtype)

    # Get unique labels from reference dataset
    labels = np.unique(y_true_ref)
    labels = sorted(labels.tolist())

    # Calculate confusion matrices
    cm_ref = confusion_matrix(y_true_ref, y_pred_ref, labels=labels)
    cm_mon = confusion_matrix(y_true_mon, y_pred_mon, labels=labels)

    # Get total counts
    total_ref = len(y_true_ref)
    total_mon = len(y_true_mon)

    # Create sample counts table
    counts_data = {
        "Dataset": ["Reference", "Monitoring"],
        "Total": [total_ref, total_mon],
    }

    # Add per-class counts
    for label in labels:
        label_str = f"Class_{label}"
        counts_data[label_str] = [
            np.sum(y_true_ref == label),
            np.sum(y_true_mon == label),
        ]

    counts_df = pd.DataFrame(counts_data)

    # Create confusion matrix metrics
    metrics = []

    if len(labels) == 2:
        # Binary classification
        tn_ref, fp_ref, fn_ref, tp_ref = cm_ref.ravel()
        tn_mon, fp_mon, fn_mon, tp_mon = cm_mon.ravel()

        confusion_elements = [
            ("True Negatives (%)", tn_ref / total_ref * 100, tn_mon / total_mon * 100),
            ("False Positives (%)", fp_ref / total_ref * 100, fp_mon / total_mon * 100),
            ("False Negatives (%)", fn_ref / total_ref * 100, fn_mon / total_mon * 100),
            ("True Positives (%)", tp_ref / total_ref * 100, tp_mon / total_mon * 100),
        ]

        for name, ref_val, mon_val in confusion_elements:
            metrics.append(
                {
                    "Metric": name,
                    "Reference": round(ref_val, 2),
                    "Monitoring": round(mon_val, 2),
                }
            )

    else:
        # Multiclass - calculate per-class metrics
        for i, label in enumerate(labels):
            # True Positives for this class
            tp_ref = cm_ref[i, i]
            tp_mon = cm_mon[i, i]

            # False Positives (sum of column minus TP)
            fp_ref = cm_ref[:, i].sum() - tp_ref
            fp_mon = cm_mon[:, i].sum() - tp_mon

            # False Negatives (sum of row minus TP)
            fn_ref = cm_ref[i, :].sum() - tp_ref
            fn_mon = cm_mon[i, :].sum() - tp_mon

            class_metrics = [
                (
                    f"True Positives_{label} (%)",
                    tp_ref / total_ref * 100,
                    tp_mon / total_mon * 100,
                ),
                (
                    f"False Positives_{label} (%)",
                    fp_ref / total_ref * 100,
                    fp_mon / total_mon * 100,
                ),
                (
                    f"False Negatives_{label} (%)",
                    fn_ref / total_ref * 100,
                    fn_mon / total_mon * 100,
                ),
            ]

            for name, ref_val, mon_val in class_metrics:
                metrics.append(
                    {
                        "Metric": name,
                        "Reference": round(ref_val, 2),
                        "Monitoring": round(mon_val, 2),
                    }
                )

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Calculate drift percentage with direction
    metrics_df["Drift (%)"] = (
        (metrics_df["Monitoring"] - metrics_df["Reference"])
        / metrics_df["Reference"].abs()
        * 100
    ).round(2)

    # Add Pass/Fail column based on absolute drift
    metrics_df["Pass/Fail"] = (
        metrics_df["Drift (%)"]
        .abs()
        .apply(lambda x: "Pass" if x < drift_pct_threshold else "Fail")
    )

    # Calculate overall pass/fail
    pass_fail_bool = (metrics_df["Pass/Fail"] == "Pass").all()

    return (
        {"Confusion Matrix Metrics": metrics_df, "Sample Counts": counts_df},
        pass_fail_bool,
    )
