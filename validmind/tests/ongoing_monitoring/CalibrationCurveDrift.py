# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from validmind import RawData, tags, tasks
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
) -> Tuple[go.Figure, Dict[str, pd.DataFrame], bool, RawData]:
    """
    Evaluates changes in probability calibration between reference and monitoring datasets.

    ### Purpose

    The Calibration Curve Drift test is designed to assess changes in the model's probability calibration
    over time. By comparing calibration curves between reference and monitoring datasets, this test helps
    identify whether the model's probability estimates remain reliable in production. This is crucial for
    understanding if the model's risk predictions maintain their intended interpretation and whether
    recalibration might be necessary.

    ### Test Mechanism

    This test proceeds by generating calibration curves for both reference and monitoring datasets. For each
    dataset, it bins the predicted probabilities and calculates the actual fraction of positives within each
    bin. It then compares these values between datasets to identify significant shifts in calibration.
    The test quantifies drift as percentage changes in both mean predicted probabilities and actual fractions
    of positives per bin, providing both visual and numerical assessments of calibration stability.

    ### Signs of High Risk

    - Large differences between reference and monitoring calibration curves
    - Systematic over-estimation or under-estimation in monitoring dataset
    - Significant drift percentages exceeding the threshold in multiple bins
    - Changes in calibration concentrated in specific probability ranges
    - Inconsistent drift patterns across the probability spectrum
    - Empty or sparse bins indicating insufficient data for reliable comparison

    ### Strengths

    - Provides visual and quantitative assessment of calibration changes
    - Identifies specific probability ranges where calibration has shifted
    - Enables early detection of systematic prediction biases
    - Includes detailed bin-by-bin comparison of calibration metrics
    - Handles edge cases with insufficient data in certain bins
    - Supports both binary and probabilistic interpretation of results

    ### Limitations

    - Requires sufficient data in each probability bin for reliable comparison
    - Sensitive to choice of number of bins and binning strategy
    - May not capture complex changes in probability distributions
    - Cannot directly suggest recalibration parameters
    - Limited to assessing probability calibration aspects
    - Results may be affected by class imbalance changes
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
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(n_bins)]

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
        RawData(
            prob_true_ref=prob_true_ref,
            prob_pred_ref=prob_pred_ref,
            prob_true_mon=prob_true_mon,
            prob_pred_mon=prob_pred_mon,
            bin_labels=bin_labels,
            model=model.input_id,
            dataset_ref=datasets[0].input_id,
            dataset_mon=datasets[1].input_id,
        ),
    )
