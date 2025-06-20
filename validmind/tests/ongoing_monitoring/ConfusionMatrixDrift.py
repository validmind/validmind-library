# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def ConfusionMatrixDrift(
    datasets: List[VMDataset], model: VMModel, drift_pct_threshold=20
) -> Tuple[Dict[str, pd.DataFrame], bool, RawData]:
    """
    Compares confusion matrix metrics between reference and monitoring datasets.

    ### Purpose

    The Confusion Matrix Drift test is designed to evaluate changes in the model's error patterns
    over time. By comparing confusion matrix elements between reference and monitoring datasets, this
    test helps identify whether the model maintains consistent prediction behavior in production. This
    is crucial for understanding if the model's error patterns have shifted and whether specific types
    of misclassifications have become more prevalent.

    ### Test Mechanism

    This test proceeds by generating confusion matrices for both reference and monitoring datasets.
    For binary classification, it tracks True Positives, True Negatives, False Positives, and False
    Negatives as percentages of total predictions. For multiclass problems, it analyzes per-class
    metrics including true positives and error rates. The test quantifies drift as percentage changes
    in these metrics between datasets, providing detailed insight into shifting prediction patterns.

    ### Signs of High Risk

    - Large drifts in confusion matrix elements exceeding threshold
    - Systematic changes in false positive or false negative rates
    - Inconsistent changes across different classes
    - Significant shifts in error patterns for specific classes
    - Unexpected improvements in certain metrics
    - Divergent trends between different types of errors

    ### Strengths

    - Provides detailed analysis of prediction behavior
    - Identifies specific types of prediction changes
    - Enables early detection of systematic errors
    - Includes comprehensive error pattern analysis
    - Supports both binary and multiclass problems
    - Maintains interpretable percentage-based metrics

    ### Limitations

    - May be sensitive to class distribution changes
    - Cannot identify root causes of prediction drift
    - Requires sufficient samples for reliable comparison
    - Limited to hard predictions (not probabilities)
    - May not capture subtle changes in decision boundaries
    - Complex interpretation for multiclass problems
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
        RawData(
            confusion_matrix_reference=cm_ref,
            confusion_matrix_monitoring=cm_mon,
            model=model.input_id,
            dataset_reference=datasets[0].input_id,
            dataset_monitoring=datasets[1].input_id,
        ),
    )
