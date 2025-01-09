# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve
from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel

from typing import List


@tags(
    "sklearn",
    "binary_classification",
    "model_performance",
    "visualization",
)
@tasks("classification", "text_classification")
def ROCCurveDrift(datasets: List[VMDataset], model: VMModel):
    """
    Compares ROC curves between reference and monitoring datasets.

    ### Purpose
    This test visualizes the differences in ROC curves and AUC scores between reference
    and monitoring datasets to identify changes in model's discriminative ability.

    ### Test Mechanism
    Generates two plots:
    1. Superimposed ROC curves for both datasets
    2. Difference between ROC curves (Monitoring - Reference)

    ### Signs of High Risk
    - Large differences between ROC curves
    - Significant drop in AUC score for monitoring dataset
    - Systematic differences in specific FPR regions
    """
    # Check for binary classification
    if len(np.unique(datasets[0].y)) > 2:
        raise SkipTestError(
            "ROC Curve Drift is only supported for binary classification models"
        )

    # Calculate ROC curves for reference dataset
    y_prob_ref = datasets[0].y_prob(model)
    y_true_ref = datasets[0].y.astype(y_prob_ref.dtype).flatten()
    fpr_ref, tpr_ref, _ = roc_curve(y_true_ref, y_prob_ref, drop_intermediate=False)
    auc_ref = roc_auc_score(y_true_ref, y_prob_ref)

    # Calculate ROC curves for monitoring dataset
    y_prob_mon = datasets[1].y_prob(model)
    y_true_mon = datasets[1].y.astype(y_prob_mon.dtype).flatten()
    fpr_mon, tpr_mon, _ = roc_curve(y_true_mon, y_prob_mon, drop_intermediate=False)
    auc_mon = roc_auc_score(y_true_mon, y_prob_mon)

    # Create superimposed ROC curves plot
    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=fpr_ref,
            y=tpr_ref,
            mode="lines",
            name=f"Reference (AUC = {auc_ref:.3f})",
            line=dict(color="blue", width=2),
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=fpr_mon,
            y=tpr_mon,
            mode="lines",
            name=f"Monitoring (AUC = {auc_mon:.3f})",
            line=dict(color="red", width=2),
        )
    )

    fig1.update_layout(
        title="ROC Curves Comparison",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        width=700,
        height=500,
    )

    # Interpolate monitoring TPR to match reference FPR points
    tpr_mon_interp = np.interp(fpr_ref, fpr_mon, tpr_mon)

    # Calculate TPR difference
    tpr_diff = tpr_mon_interp - tpr_ref

    # Create difference plot
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=fpr_ref,
            y=tpr_diff,
            mode="lines",
            name="TPR Difference",
            line=dict(color="purple", width=2),
        )
    )

    # Add horizontal line at y=0
    fig2.add_hline(y=0, line=dict(color="grey", dash="dash"), name="No Difference")

    fig2.update_layout(
        title="ROC Curve Difference (Monitoring - Reference)",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="TPR Difference"),
        width=700,
        height=500,
    )

    return fig1, fig2
