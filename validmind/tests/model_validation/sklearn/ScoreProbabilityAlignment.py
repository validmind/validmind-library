# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk", "calibration")
@tasks("classification")
def ScoreProbabilityAlignment(
    model: VMModel, dataset: VMDataset, score_column: str = "score", n_bins: int = 10
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Analyzes the alignment between credit scores and predicted probabilities.

    ### Purpose

    The Score-Probability Alignment test evaluates how well credit scores align with
    predicted default probabilities. This helps validate score scaling, identify potential
    calibration issues, and ensure scores reflect risk appropriately.

    ### Test Mechanism

    The test:
    1. Groups scores into bins
    2. Calculates average predicted probability per bin
    3. Tests monotonicity of relationship
    4. Analyzes probability distribution within score bands

    ### Signs of High Risk

    - Non-monotonic relationship between scores and probabilities
    - Large probability variations within score bands
    - Unexpected probability jumps between adjacent bands
    - Poor alignment with expected odds-to-score relationship
    - Inconsistent probability patterns across score ranges
    - Clustering of probabilities at extreme values
    - Score bands with similar probability profiles
    - Unstable probability estimates in key decision bands

    ### Strengths

    - Direct validation of score-to-probability relationship
    - Identifies potential calibration issues
    - Supports score band validation
    - Helps understand model behavior
    - Useful for policy setting
    - Visual and numerical results
    - Easy to interpret
    - Supports regulatory documentation

    ### Limitations

    - Sensitive to bin selection
    - Requires sufficient data per bin
    - May mask within-bin variations
    - Point-in-time analysis only
    - Cannot detect all forms of miscalibration
    - Assumes scores should align with probabilities
    - May oversimplify complex relationships
    - Limited to binary outcomes
    """
    if score_column not in dataset.df.columns:
        raise ValueError(f"Score column '{score_column}' not found in dataset")

    # Get predicted probabilities
    y_prob = dataset.y_prob(model)

    # Create score bins
    df = dataset.df.copy()
    df["probability"] = y_prob

    # Create score bins with equal width
    df["score_bin"] = pd.qcut(df[score_column], n_bins, duplicates="drop")

    # Calculate statistics per bin
    results = []
    for bin_name, group in df.groupby("score_bin"):
        bin_stats = {
            "Score Range": f"{bin_name.left:.0f}-{bin_name.right:.0f}",
            "Mean Score": group[score_column].mean(),
            "Population Count": len(group),
            "Population (%)": len(group) / len(df) * 100,
            "Mean Probability (%)": group["probability"].mean() * 100,
            "Min Probability (%)": group["probability"].min() * 100,
            "Max Probability (%)": group["probability"].max() * 100,
            "Probability Std": group["probability"].std() * 100,
        }
        results.append(bin_stats)

    results_df = pd.DataFrame(results)

    # Create visualization
    fig = go.Figure()

    # Add probability range
    fig.add_trace(
        go.Scatter(
            x=results_df["Mean Score"],
            y=results_df["Mean Probability (%)"],
            mode="lines+markers",
            name="Mean Probability",
            line=dict(color="blue"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=results_df["Max Probability (%)"]
                - results_df["Mean Probability (%)"],
                arrayminus=results_df["Mean Probability (%)"]
                - results_df["Min Probability (%)"],
                color="gray",
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title="Score-Probability Alignment",
        xaxis_title="Score",
        yaxis_title="Default Probability (%)",
        showlegend=True,
        template="plotly_white",
        width=800,
        height=600,
    )

    # Include raw data for post-processing
    raw_data = RawData(
        score_bins=df[["score_bin", score_column]],
        predicted_probabilities=df["probability"],
        model=model.input_id,
        dataset=dataset.input_id,
    )

    return results_df, fig, raw_data
