# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization")
@tasks("monitoring")
def PredictionQuantilesAcrossFeatures(
    datasets: List[VMDataset], model: VMModel
) -> Tuple[go.Figure, ...]:
    """
    Assesses differences in model prediction distributions across individual features between reference
    and monitoring datasets through quantile analysis.

    ### Purpose

    This test aims to visualize how prediction distributions vary across feature values by showing
    quantile information between reference and monitoring datasets. It helps identify significant
    shifts in prediction patterns and potential areas of model instability.

    ### Test Mechanism

    The test generates box plots for each feature, comparing prediction probability distributions
    between the reference and monitoring datasets. Each plot consists of two subplots showing the
    quantile distribution of predictions: one for reference data and one for monitoring data.

    ### Signs of High Risk

    - Significant differences in prediction distributions between reference and monitoring data
    - Unexpected shifts in prediction quantiles across feature values
    - Large changes in prediction variability between datasets

    ### Strengths

    - Provides clear visualization of prediction distribution changes
    - Shows outliers and variability in predictions across features
    - Enables quick identification of problematic feature ranges

    ### Limitations

    - May not capture complex relationships between features and predictions
    - Quantile analysis may smooth over important individual predictions
    - Requires careful interpretation of distribution changes
    """

    feature_columns = datasets[0].feature_columns
    y_prob_reference = datasets[0].y_prob(model)
    y_prob_monitoring = datasets[1].y_prob(model)

    figures_to_save = []
    for column in feature_columns:
        # Create subplot
        fig = make_subplots(1, 2, subplot_titles=("Reference", "Monitoring"))

        # Add reference box plot
        fig.add_trace(
            go.Box(
                x=datasets[0].df[column],
                y=y_prob_reference,
                name="Reference",
                boxpoints="outliers",
                marker_color="blue",
            ),
            row=1,
            col=1,
        )

        # Add monitoring box plot
        fig.add_trace(
            go.Box(
                x=datasets[1].df[column],
                y=y_prob_monitoring,
                name="Monitoring",
                boxpoints="outliers",
                marker_color="red",
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"Prediction Distributions vs {column}",
            showlegend=False,
            width=800,
            height=400,
        )

        # Update axes
        fig.update_xaxes(title=column)
        fig.update_yaxes(title="Prediction Value")

        figures_to_save.append(fig)

    return tuple(figures_to_save)
