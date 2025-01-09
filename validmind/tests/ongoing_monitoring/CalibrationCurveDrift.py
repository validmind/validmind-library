# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from typing import List
from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn",
    "binary_classification",
    "model_performance",
    "visualization",
)
@tasks("classification", "text_classification")
def CalibrationCurveDrift(
    datasets: List[VMDataset],
    model: VMModel,
    n_bins: int = 10,
    drift_pct_threshold: float = 20,
):
    """
    Compares calibration curves between reference and monitoring datasets.

    ### Purpose
    This test visualizes and quantifies differences in probability calibration between
    reference and monitoring datasets to identify changes in model's probability estimates.

    ### Test Mechanism
    Generates a plot with superimposed calibration curves and two tables comparing:
    1. Mean predicted probabilities per bin
    2. Actual fraction of positives per bin

    ### Signs of High Risk
    - Large differences between calibration curves
    - Systematic over/under-estimation in monitoring dataset
    - Changes in calibration for specific probability ranges
    """
    # Check for binary classification
    if len(np.unique(datasets[0].y)) > 2:
        raise SkipTestError(
            "Calibration Curve Drift is only supported for binary classification models"
        )

    # Calculate calibration for reference dataset
    y_prob_ref = datasets[0].y_prob(model)
    y_true_ref = datasets[0].y.astype(y_prob_ref.dtype).flatten()
    prob_true_ref, prob_pred_ref = calibration_curve(
        y_true_ref, y_prob_ref, n_bins=n_bins, strategy="uniform"
    )

    # Calculate calibration for monitoring dataset
    y_prob_mon = datasets[1].y_prob(model)
    y_true_mon = datasets[1].y.astype(y_prob_mon.dtype).flatten()
    prob_true_mon, prob_pred_mon = calibration_curve(
        y_true_mon, y_prob_mon, n_bins=n_bins, strategy="uniform"
    )

    # Create bin labels
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]

    # Create predicted probabilities table
    pred_metrics = []
    for i in range(n_bins):
        ref_val = "no data" if i >= len(prob_pred_ref) else round(prob_pred_ref[i], 3)
        mon_val = "no data" if i >= len(prob_pred_mon) else round(prob_pred_mon[i], 3)

        pred_metrics.append(
            {"Bin": bin_labels[i], "Reference": ref_val, "Monitoring": mon_val}
        )

    pred_df = pd.DataFrame(pred_metrics)

    # Calculate drift only for bins with data
    mask = (pred_df["Reference"] != "no data") & (pred_df["Monitoring"] != "no data")
    pred_df["Drift (%)"] = None
    pred_df.loc[mask, "Drift (%)"] = (
        (
            pd.to_numeric(pred_df.loc[mask, "Monitoring"])
            - pd.to_numeric(pred_df.loc[mask, "Reference"])
        )
        / pd.to_numeric(pred_df.loc[mask, "Reference"]).abs()
        * 100
    ).round(2)

    pred_df["Pass/Fail"] = None
    pred_df.loc[mask, "Pass/Fail"] = (
        pred_df.loc[mask, "Drift (%)"]
        .abs()
        .apply(lambda x: "Pass" if x < drift_pct_threshold else "Fail")
    )
    pred_df.loc[~mask, "Pass/Fail"] = "N/A"

    # Create fraction of positives table
    true_metrics = []
    for i in range(n_bins):
        ref_val = "no data" if i >= len(prob_true_ref) else round(prob_true_ref[i], 3)
        mon_val = "no data" if i >= len(prob_true_mon) else round(prob_true_mon[i], 3)

        true_metrics.append(
            {"Bin": bin_labels[i], "Reference": ref_val, "Monitoring": mon_val}
        )

    true_df = pd.DataFrame(true_metrics)

    # Calculate drift only for bins with data
    mask = (true_df["Reference"] != "no data") & (true_df["Monitoring"] != "no data")
    true_df["Drift (%)"] = None
    true_df.loc[mask, "Drift (%)"] = (
        (
            pd.to_numeric(true_df.loc[mask, "Monitoring"])
            - pd.to_numeric(true_df.loc[mask, "Reference"])
        )
        / pd.to_numeric(true_df.loc[mask, "Reference"]).abs()
        * 100
    ).round(2)

    true_df["Pass/Fail"] = None
    true_df.loc[mask, "Pass/Fail"] = (
        true_df.loc[mask, "Drift (%)"]
        .abs()
        .apply(lambda x: "Pass" if x < drift_pct_threshold else "Fail")
    )
    true_df.loc[~mask, "Pass/Fail"] = "N/A"

    # Create figure
    fig = go.Figure()

    # Add perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="grey", dash="dash"),
        )
    )

    # Add reference calibration curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred_ref,
            y=prob_true_ref,
            mode="lines+markers",
            name="Reference",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        )
    )

    # Add monitoring calibration curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred_mon,
            y=prob_true_mon,
            mode="lines+markers",
            name="Monitoring",
            line=dict(color="red", width=2),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="Calibration Curves Comparison",
        xaxis=dict(title="Mean Predicted Probability", range=[0, 1]),
        yaxis=dict(title="Fraction of Positives", range=[0, 1]),
        width=700,
        height=500,
    )

    # Calculate overall pass/fail (only for bins with data)
    pass_fail_bool = (pred_df.loc[mask, "Pass/Fail"] == "Pass").all() and (
        true_df.loc[mask, "Pass/Fail"] == "Pass"
    ).all()

    return (
        fig,
        {"Mean Predicted Probabilities": pred_df, "Fraction of Positives": true_df},
        pass_fail_bool,
    )
