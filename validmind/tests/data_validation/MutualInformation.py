# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from validmind import tags, tasks
from validmind.vm_models import VMDataset
from validmind.vm_models.result import RawData


@tags("feature_selection", "data_analysis")
@tasks("classification", "regression")
def MutualInformation(
    dataset: VMDataset, min_threshold: float = 0.01, task: str = "classification"
) -> Tuple[go.Figure, RawData]:
    """
    Calculates mutual information scores between features and target variable to evaluate feature relevance.

    ### Purpose

    The Mutual Information test quantifies the predictive power of each feature by measuring its statistical
    dependency with the target variable. This helps identify relevant features for model training and
    detect potential redundant or irrelevant variables, supporting feature selection decisions and model
    interpretability.

    ### Test Mechanism

    The test employs sklearn's mutual_info_classif/mutual_info_regression functions to compute mutual
    information between each feature and the target. It produces a normalized score (0 to 1) for each
    feature, where higher scores indicate stronger relationships. Results are presented in both tabular
    format and visualized through a bar plot with a configurable threshold line.

    ### Signs of High Risk

    - Many features showing very low mutual information scores
    - Key business features exhibiting unexpectedly low scores
    - All features showing similar, low information content
    - Large discrepancy between business importance and MI scores
    - Highly skewed distribution of MI scores
    - Critical features below the minimum threshold
    - Unexpected zero or near-zero scores for known important features
    - Inconsistent scores across different data samples

    ### Strengths

    - Captures non-linear relationships between features and target
    - Scale-invariant measurement of feature relevance
    - Works for both classification and regression tasks
    - Provides interpretable scores (0 to 1 scale)
    - Supports automated feature selection
    - No assumptions about data distribution
    - Handles numerical and categorical features
    - Computationally efficient for most datasets

    ### Limitations

    - Requires sufficient data for reliable estimates
    - May be computationally intensive for very large datasets
    - Cannot detect redundant features (pairwise relationships)
    - Sensitive to feature discretization for continuous variables
    - Does not account for feature interactions
    - May underestimate importance of rare but crucial events
    - Cannot handle missing values directly
    - May be affected by extreme class imbalance
    """
    if task not in ["classification", "regression"]:
        raise ValueError("task must be either 'classification' or 'regression'")

    # Check if numeric features exist
    if not dataset.feature_columns_numeric:
        raise ValueError(
            "No numeric features found in dataset. Mutual Information test requires numeric features."
        )

    # Check if target column exists
    if not dataset.target_column:
        raise ValueError(
            "Target column is required for Mutual Information calculation but was not provided."
        )

    X = dataset._df[dataset.feature_columns_numeric]
    y = dataset._df[dataset.target_column]

    # Select appropriate MI function based on task type
    if task == "classification":
        mi_scores = mutual_info_classif(X, y)
    else:
        mi_scores = mutual_info_regression(X, y)

    # Create Plotly figure
    fig = go.Figure()

    # Sort data for better visualization
    sorted_indices = sorted(
        range(len(mi_scores)), key=lambda k: mi_scores[k], reverse=True
    )
    sorted_features = [dataset.feature_columns[i] for i in sorted_indices]
    sorted_scores = [mi_scores[i] for i in sorted_indices]

    # Add bar plot
    fig.add_trace(
        go.Bar(
            x=sorted_features,
            y=sorted_scores,
            marker_color=[
                "blue" if score >= min_threshold else "red" for score in sorted_scores
            ],
            name="Mutual Information Score",
        )
    )

    # Add threshold line
    fig.add_hline(
        y=min_threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold ({min_threshold})",
        annotation_position="right",
    )

    # Update layout
    fig.update_layout(
        title="Mutual Information Scores by Feature",
        xaxis_title="Features",
        yaxis_title="Mutual Information Score",
        xaxis_tickangle=-45,
        showlegend=False,
        width=1000,
        height=600,
        template="plotly_white",
    )

    return fig, RawData(
        mutual_information_scores={
            feature: score for feature, score in zip(sorted_features, sorted_scores)
        },
        dataset=dataset.input_id,
    )
