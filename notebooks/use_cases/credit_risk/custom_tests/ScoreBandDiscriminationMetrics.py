# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Tuple
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk", "scorecard")
@tasks("classification")
def ScoreBandDiscriminationMetrics(
    dataset: VMDataset,
    model: VMModel,
    score_column: str = "score",
    score_bands: list = None,
    title: str = "Score Band Discrimination Metrics",
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Evaluates discrimination metrics (AUC, GINI, KS) across different score bands for credit risk assessment.

    ### Purpose

    The Score Band Discrimination Metrics test is designed to evaluate the model's discriminatory power across
    different score ranges. By segmenting the score distribution into bands and calculating key discrimination
    metrics within each band, this test helps identify whether the model maintains consistent performance across
    the entire score spectrum. This is crucial for understanding if the model's ability to separate good and bad
    accounts varies significantly across different score ranges.

    ### Test Mechanism

    This test proceeds by first segmenting the score distribution into predefined bands. For each band, it
    calculates three key discrimination metrics: AUC (Area Under the Curve), GINI coefficient, and KS
    (Kolmogorov-Smirnov) statistic. The AUC measures the model's ability to rank order risk, the GINI
    coefficient provides a measure of inequality in the predictions, and the KS statistic quantifies the maximum
    separation between cumulative distributions. The test also tracks the population distribution and default
    rates across bands to provide context for the discrimination metrics.

    ### Signs of High Risk

    - Significant variations in discrimination metrics between adjacent score bands
    - Very low metric values in specific score ranges, indicating poor discrimination
    - Inconsistent patterns in metric values across the score spectrum
    - Large disparities between band-specific metrics and overall metrics
    - Unexpected relationships between default rates and discrimination metrics
    - Insufficient population in certain score bands for reliable metric calculation

    ### Strengths

    - Provides a comprehensive view of model discrimination across the score spectrum
    - Combines multiple complementary metrics for robust performance assessment
    - Identifies specific score ranges where model performance might be suboptimal
    - Includes population and default rate context for better interpretation
    - Handles edge cases such as single-class bands and insufficient data
    - Enables visual comparison of metrics across score bands

    ### Limitations

    - Requires sufficient data in each score band for reliable metric calculation
    - May be sensitive to the choice of score band boundaries
    - Does not account for business importance of different score ranges
    - Metrics may be unstable in bands with very low default rates
    - Cannot directly suggest optimal score band boundaries
    - Limited to assessing discrimination aspects of model performance
    """
    if score_column not in dataset.df.columns:
        raise ValueError(f"Score column '{score_column}' not found in dataset")

    df = dataset.df.copy()

    # Default score bands if none provided
    if score_bands is None:
        score_bands = [410, 440, 470]

    # Create band labels
    band_labels = [
        f"{score_bands[i]}-{score_bands[i+1]}" for i in range(len(score_bands) - 1)
    ]
    band_labels.insert(0, f"<{score_bands[0]}")
    band_labels.append(f">{score_bands[-1]}")

    # Bin the scores
    df["score_band"] = pd.cut(
        df[score_column], bins=[-np.inf] + score_bands + [np.inf], labels=band_labels
    )

    # Calculate metrics for each band
    results = []
    for band in band_labels:
        band_mask = df["score_band"] == band
        if band_mask.sum() > 1:  # Need at least 2 samples
            y_true = df[band_mask][dataset.target_column].values
            y_prob = dataset.y_prob(model)[
                band_mask
            ]  # Get predicted probabilities using dataset method

            # Convert to float arrays
            y_true = np.array(y_true, dtype=float)
            y_prob = np.array(y_prob, dtype=float)

            # Calculate metrics
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                ks = max(tpr - fpr)
                auc = roc_auc_score(y_true, y_prob)
                gini = 2 * auc - 1
            except ValueError:  # Handle cases with single class
                ks, auc, gini = 0, 0.5, 0

            results.append(
                {
                    "Score Band": band,
                    "Population Count": band_mask.sum(),
                    "Population (%)": (band_mask.sum() / len(df)) * 100,
                    "AUC": auc,
                    "GINI": gini,
                    "KS": ks,
                    "Default Rate (%)": (y_true.mean() * 100),
                }
            )

    # Calculate total metrics
    y_true = df[dataset.target_column].values
    y_prob = dataset.y_prob(model)  # Get predicted probabilities for total calculation

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    total_ks = max(tpr - fpr)
    total_auc = roc_auc_score(y_true, y_prob)
    total_gini = 2 * total_auc - 1

    # Add total row
    results.append(
        {
            "Score Band": f"Total ({df[score_column].min():.0f}-{df[score_column].max():.0f})",
            "Population Count": len(df),
            "Population (%)": 100.0,
            "AUC": total_auc,
            "GINI": total_gini,
            "KS": total_ks,
            "Default Rate (%)": (y_true.mean() * 100),
        }
    )

    results_df = pd.DataFrame(results)

    # Create visualization (excluding total)
    fig = go.Figure()

    # Filter out the total row for plotting
    plot_df = results_df[results_df["Score Band"].str.contains("Total") == False]

    # Add metric bars
    for metric, color in [
        ("AUC", "rgb(31, 119, 180)"),
        ("GINI", "rgb(255, 127, 14)"),
        ("KS", "rgb(44, 160, 44)"),
    ]:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=plot_df["Score Band"],
                y=plot_df[metric],
                marker_color=color,
            )
        )

    # Add default rate line (excluding total)
    fig.add_trace(
        go.Scatter(
            name="Default Rate (%)",
            x=plot_df["Score Band"],
            y=plot_df["Default Rate (%)"],
            yaxis="y2",
            line=dict(color="red", width=2),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Score Band",
        yaxis_title="Discrimination Metrics",
        yaxis2=dict(title="Default Rate (%)", overlaying="y", side="right"),
        barmode="group",
        showlegend=True,
        height=600,
    )

    return fig, results_df
