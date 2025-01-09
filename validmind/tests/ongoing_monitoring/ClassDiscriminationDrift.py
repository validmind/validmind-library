# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from scipy import stats
from typing import List
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    return roc_auc_score(lb.transform(y_test), lb.transform(y_pred), average=average)


def calculate_gini(y_true, y_prob):
    """Calculate Gini coefficient (2*AUC - 1)"""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def calculate_ks_statistic(y_true, y_prob):
    """Calculate Kolmogorov-Smirnov statistic"""
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    return stats.ks_2samp(pos_scores, neg_scores).statistic


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def ClassDiscriminationDrift(
    datasets: List[VMDataset], model: VMModel, drift_pct_threshold=20
):
    """
    Compares classification discrimination metrics between reference and monitoring datasets.

    ### Purpose
    This test evaluates drift in discrimination metrics including ROC AUC, GINI, and KS statistics.

    ### Test Mechanism
    Calculates discrimination metrics for both reference and monitoring datasets and
    compares them to identify significant changes in model's discriminative power.

    ### Signs of High Risk
    - Large drifts in discrimination metrics (above threshold)
    - Significant drops in ROC AUC or GINI coefficient
    - Reduced class separation as indicated by KS statistic
    """
    # Get predictions and true values
    y_true_ref = datasets[0].y
    y_true_mon = datasets[1].y

    metrics = []

    # Handle binary vs multiclass
    if len(np.unique(y_true_ref)) == 2:
        # Binary classification
        y_prob_ref = datasets[0].y_prob(model)
        y_prob_mon = datasets[1].y_prob(model)

        # ROC AUC
        roc_auc_ref = roc_auc_score(y_true_ref, y_prob_ref)
        roc_auc_mon = roc_auc_score(y_true_mon, y_prob_mon)
        metrics.append(
            {"Metric": "ROC_AUC", "Reference": roc_auc_ref, "Monitoring": roc_auc_mon}
        )

        # GINI
        gini_ref = calculate_gini(y_true_ref, y_prob_ref)
        gini_mon = calculate_gini(y_true_mon, y_prob_mon)
        metrics.append(
            {"Metric": "GINI", "Reference": gini_ref, "Monitoring": gini_mon}
        )

        # KS Statistic
        ks_ref = calculate_ks_statistic(y_true_ref, y_prob_ref)
        ks_mon = calculate_ks_statistic(y_true_mon, y_prob_mon)
        metrics.append(
            {"Metric": "KS_Statistic", "Reference": ks_ref, "Monitoring": ks_mon}
        )

    else:
        # Multiclass
        y_pred_ref = datasets[0].y_pred(model)
        y_pred_mon = datasets[1].y_pred(model)

        # Only ROC AUC for multiclass
        roc_auc_ref = multiclass_roc_auc_score(y_true_ref, y_pred_ref)
        roc_auc_mon = multiclass_roc_auc_score(y_true_mon, y_pred_mon)
        metrics.append(
            {
                "Metric": "ROC_AUC_Macro",
                "Reference": roc_auc_ref,
                "Monitoring": roc_auc_mon,
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

    return ({"Classification Discrimination Metrics": df}, pass_fail_bool)
