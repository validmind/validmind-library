# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset

try:
    from scipy import stats
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for ScorecardHistogramDrift. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e


@tags("visualization", "credit_risk", "logistic_regression")
@tasks("classification")
def ScorecardHistogramDrift(
    datasets: List[VMDataset],
    score_column: str = "score",
    title: str = "Scorecard Histogram Drift",
    drift_pct_threshold: float = 20.0,
) -> Tuple[go.Figure, Dict[str, pd.DataFrame], bool]:
    """
    Compares score distributions between reference and monitoring datasets for each class.

    ### Purpose

    The Scorecard Histogram Drift test is designed to evaluate changes in the model's scoring
    patterns over time. By comparing score distributions between reference and monitoring datasets
    for each class, this test helps identify whether the model's scoring behavior remains stable
    in production. This is crucial for understanding if the model's risk assessment maintains
    consistent patterns and whether specific score ranges have experienced significant shifts
    in their distribution.

    ### Test Mechanism

    This test proceeds by generating histograms of scores for each class in both reference and
    monitoring datasets. It analyzes distribution characteristics through multiple statistical
    moments: mean, variance, skewness, and kurtosis. The test quantifies drift as percentage
    changes in these moments between datasets, providing both visual and numerical assessments
    of distribution stability. Special attention is paid to class-specific distribution changes.

    ### Signs of High Risk

    - Significant shifts in score distribution shapes
    - Large drifts in distribution moments exceeding threshold
    - Changes in the relative positioning of class distributions
    - Appearance of new modes or peaks in monitoring data
    - Unexpected changes in score spread or concentration
    - Systematic shifts in class-specific scoring patterns

    ### Strengths

    - Provides class-specific distribution analysis
    - Identifies detailed changes in scoring patterns
    - Enables visual comparison of distributions
    - Includes comprehensive moment analysis
    - Supports multiple class evaluation
    - Maintains interpretable score scale

    ### Limitations

    - Sensitive to binning choices in visualization
    - Requires sufficient samples per class
    - Cannot suggest score adjustments
    - May not capture subtle distribution changes
    - Complex interpretation with multiple classes
    - Limited to univariate score analysis
    """
    # Verify score column exists
    if score_column not in datasets[0].df.columns:
        raise ValueError(
            f"Score column '{score_column}' not found in reference dataset"
        )
    if score_column not in datasets[1].df.columns:
        raise ValueError(
            f"Score column '{score_column}' not found in monitoring dataset"
        )

    # Get reference and monitoring data
    df_ref = datasets[0].df
    df_mon = datasets[1].df

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
        # Get scores for current class
        ref_scores = df_ref[df_ref[datasets[0].target_column] == class_value][
            score_column
        ]
        mon_scores = df_mon[df_mon[datasets[1].target_column] == class_value][
            score_column
        ]

        # Calculate distribution moments
        ref_stats = {
            "Mean": np.mean(ref_scores),
            "Variance": np.var(ref_scores),
            "Skewness": stats.skew(ref_scores),
            "Kurtosis": stats.kurtosis(ref_scores),
        }

        mon_stats = {
            "Mean": np.mean(mon_scores),
            "Variance": np.var(mon_scores),
            "Skewness": stats.skew(mon_scores),
            "Kurtosis": stats.kurtosis(mon_scores),
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
                x=ref_scores,
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
                x=mon_scores,
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
        fig.update_xaxes(title_text="Score", row=i + 1, col=1)
        fig.update_yaxes(title_text="Frequency", row=i + 1, col=1)

        # Add separate legend for each subplot
        fig.update_layout(
            **{
                f'legend{i + 1 if i > 0 else ""}': dict(
                    yanchor="middle",
                    y=1 - (i / len(classes)) - (0.5 / len(classes)),
                    xanchor="left",
                    x=1.05,
                    tracegroupgap=5,
                )
            }
        )

    return fig, tables, all_passed
