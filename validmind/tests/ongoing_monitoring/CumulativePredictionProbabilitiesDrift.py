# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk")
@tasks("classification")
def CumulativePredictionProbabilitiesDrift(
    datasets: List[VMDataset],
    model: VMModel,
) -> Tuple[go.Figure, RawData]:
    """
    Compares cumulative prediction probability distributions between reference and monitoring datasets.

    ### Purpose

    The Cumulative Prediction Probabilities Drift test is designed to evaluate changes in the model's
    probability predictions over time. By comparing cumulative distribution functions of predicted
    probabilities between reference and monitoring datasets, this test helps identify whether the
    model's probability assignments remain stable in production. This is crucial for understanding if
    the model's risk assessment behavior has shifted and whether its probability calibration remains
    consistent.

    ### Test Mechanism

    This test proceeds by generating cumulative distribution functions (CDFs) of predicted probabilities
    for both reference and monitoring datasets. For each class, it plots the cumulative proportion of
    predictions against probability values, enabling direct comparison of probability distributions.
    The test visualizes both the CDFs and their differences, providing insight into how probability
    assignments have shifted across the entire probability range.

    ### Signs of High Risk

    - Large gaps between reference and monitoring CDFs
    - Systematic shifts in probability assignments
    - Concentration of differences in specific probability ranges
    - Changes in the shape of probability distributions
    - Unexpected patterns in cumulative differences
    - Significant shifts in probability thresholds

    ### Strengths

    - Provides comprehensive view of probability changes
    - Identifies specific probability ranges with drift
    - Enables visualization of distribution differences
    - Supports analysis across multiple classes
    - Maintains interpretable probability scale
    - Captures subtle changes in probability assignments

    ### Limitations

    - Does not provide single drift metric
    - May be complex to interpret for multiple classes
    - Cannot suggest probability recalibration
    - Requires visual inspection for assessment
    - Sensitive to sample size differences
    - May not capture class-specific calibration issues
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

    # Define colors
    ref_color = "rgba(31, 119, 180, 0.8)"  # Blue with 0.8 opacity
    mon_color = "rgba(255, 127, 14, 0.8)"  # Orange with 0.8 opacity
    diff_color = "rgba(148, 103, 189, 0.8)"  # Purple with 0.8 opacity

    figures = []
    raw_data = {}
    for class_value in classes:
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                f"Cumulative Distributions - Class {class_value}",
                "Difference (Monitoring - Reference)",
            ],
            vertical_spacing=0.15,
            shared_xaxes=True,
        )

        # Get probabilities for current class
        ref_probs = df_ref[df_ref[datasets[0].target_column] == class_value][
            "probabilities"
        ]
        mon_probs = df_mon[df_mon[datasets[1].target_column] == class_value][
            "probabilities"
        ]

        # Calculate cumulative distributions
        ref_sorted = np.sort(ref_probs)
        ref_cumsum = np.arange(len(ref_sorted)) / float(len(ref_sorted))

        mon_sorted = np.sort(mon_probs)
        mon_cumsum = np.arange(len(mon_sorted)) / float(len(mon_sorted))

        # Reference dataset cumulative curve
        fig.add_trace(
            go.Scatter(
                x=ref_sorted,
                y=ref_cumsum,
                mode="lines",
                name="Reference",
                line=dict(color=ref_color, width=2),
            ),
            row=1,
            col=1,
        )

        # Monitoring dataset cumulative curve
        fig.add_trace(
            go.Scatter(
                x=mon_sorted,
                y=mon_cumsum,
                mode="lines",
                name="Monitoring",
                line=dict(color=mon_color, width=2),
            ),
            row=1,
            col=1,
        )

        # Calculate and plot difference
        # Interpolate monitoring values to match reference x-points
        mon_interp = np.interp(ref_sorted, mon_sorted, mon_cumsum)
        difference = mon_interp - ref_cumsum

        fig.add_trace(
            go.Scatter(
                x=ref_sorted,
                y=difference,
                mode="lines",
                name="Difference",
                line=dict(color=diff_color, width=2),
            ),
            row=2,
            col=1,
        )

        # Add horizontal line at y=0 for difference plot
        fig.add_hline(y=0, line=dict(color="grey", dash="dash"), row=2, col=1)

        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            showlegend=True,
            legend=dict(yanchor="middle", y=0.9, xanchor="left", x=1.05),
        )

        # Update axes
        fig.update_xaxes(title_text="Probability", range=[0, 1], row=2, col=1)
        fig.update_xaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(
            title_text="Cumulative Distribution", range=[0, 1], row=1, col=1
        )
        fig.update_yaxes(title_text="Difference", row=2, col=1)

        figures.append(fig)

        # Store raw data for current class
        raw_data[f"class_{class_value}_ref_probs"] = ref_probs
        raw_data[f"class_{class_value}_mon_probs"] = mon_probs
        raw_data[f"class_{class_value}_ref_sorted"] = ref_sorted
        raw_data[f"class_{class_value}_ref_cumsum"] = ref_cumsum
        raw_data[f"class_{class_value}_mon_sorted"] = mon_sorted
        raw_data[f"class_{class_value}_mon_cumsum"] = mon_cumsum

    return tuple(figures) + (
        RawData(
            model=model.input_id,
            dataset_reference=datasets[0].input_id,
            dataset_monitoring=datasets[1].input_id,
            **raw_data,
        ),
    )
