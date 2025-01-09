# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import List
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk")
@tasks("classification")
def PredictionProbabilitiesHistogramDrift(
    datasets: List[VMDataset],
    model: VMModel,
    title="Prediction Probabilities Histogram Drift",
    drift_pct_threshold: float = 20.0,
):
    """
    Compares prediction probability distributions between reference and monitoring datasets
    for each class.

    ### Purpose
    This test visualizes and quantifies changes in the model's probability predictions
    between reference and monitoring datasets by comparing their distributions for each class.

    ### Test Mechanism
    - Creates histograms of prediction probabilities for each class
    - Superimposes reference and monitoring distributions
    - Computes distribution moments and their drift
    - Uses separate subplots for each class for clear comparison

    ### Signs of High Risk
    - Significant shifts in probability distributions
    - Changes in the shape of distributions
    - New modes or peaks appearing in monitoring data
    - Large differences in distribution moments
    """
    # Get predictions and true values
    y_prob_ref = datasets[0].y_prob(model)
    df_ref = datasets[0].df.copy()
    df_ref["probabilities"] = y_prob_ref

    y_prob_mon = datasets[1].y_prob(model)
    df_mon = datasets[1].df.copy()
    df_mon["probabilities"] = y_prob_mon

    # Get unique classes
    classes = sorted(df_ref[datasets[0].target_column].unique())

    # Create subplots with more horizontal space for legends
    fig = make_subplots(
        rows=len(classes),
        cols=1,
        subplot_titles=[f"Class {cls}" for cls in classes],
        horizontal_spacing=0.15,
    )

    # Define colors
    ref_color = "rgba(31, 119, 180, 0.8)"  # Blue with 0.8 opacity
    mon_color = "rgba(255, 127, 14, 0.8)"  # Orange with 0.8 opacity

    # Dictionary to store tables for each class
    tables = {}
    all_passed = True  # Track overall pass/fail

    # Add histograms and create tables for each class
    for i, class_value in enumerate(classes, start=1):
        # Get probabilities for current class
        ref_probs = df_ref[df_ref[datasets[0].target_column] == class_value][
            "probabilities"
        ]
        mon_probs = df_mon[df_mon[datasets[1].target_column] == class_value][
            "probabilities"
        ]

        # Calculate distribution moments
        ref_stats = {
            "Mean": np.mean(ref_probs),
            "Variance": np.var(ref_probs),
            "Skewness": stats.skew(ref_probs),
            "Kurtosis": stats.kurtosis(ref_probs),
        }

        mon_stats = {
            "Mean": np.mean(mon_probs),
            "Variance": np.var(mon_probs),
            "Skewness": stats.skew(mon_probs),
            "Kurtosis": stats.kurtosis(mon_probs),
        }

        # Create table for this class
        table_data = []
        class_passed = True  # Track pass/fail for this class

        for stat_name in ["Mean", "Variance", "Skewness", "Kurtosis"]:
            ref_val = ref_stats[stat_name]
            mon_val = mon_stats[stat_name]
            drift = (
                ((mon_val - ref_val) / abs(ref_val)) * 100 if ref_val != 0 else np.inf
            )
            passed = abs(drift) < drift_pct_threshold
            class_passed &= passed  # Update class pass/fail

            table_data.append(
                {
                    "Statistic": stat_name,
                    "Reference": round(ref_val, 4),
                    "Monitoring": round(mon_val, 4),
                    "Drift (%)": round(drift, 2),
                    "Pass/Fail": "Pass" if passed else "Fail",
                }
            )

        tables[f"Class {class_value}"] = pd.DataFrame(table_data)
        all_passed &= class_passed  # Update overall pass/fail

        # Reference dataset histogram
        fig.add_trace(
            go.Histogram(
                x=ref_probs,
                name=f"Reference - Class {class_value}",
                marker_color=ref_color,
                showlegend=True,
                legendrank=i * 2 - 1,
            ),
            row=i,
            col=1,
        )

        # Monitoring dataset histogram
        fig.add_trace(
            go.Histogram(
                x=mon_probs,
                name=f"Monitoring - Class {class_value}",
                marker_color=mon_color,
                showlegend=True,
                legendrank=i * 2,
            ),
            row=i,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title_text=title,
        barmode="overlay",
        height=300 * len(classes),
        width=1000,
        showlegend=True,
    )

    # Update axes labels and add separate legends for each subplot
    for i in range(len(classes)):
        fig.update_xaxes(title_text="Probability", row=i + 1, col=1)
        fig.update_yaxes(title_text="Frequency", row=i + 1, col=1)

        # Add separate legend for each subplot
        fig.update_layout(
            **{
                f'legend{i+1 if i > 0 else ""}': dict(
                    yanchor="middle",
                    y=1 - (i / len(classes)) - (0.5 / len(classes)),
                    xanchor="left",
                    x=1.05,
                    tracegroupgap=5,
                )
            }
        )

    return fig, tables, all_passed
