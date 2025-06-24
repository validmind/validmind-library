# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import numpy as np
import plotly.graph_objects as go

from validmind import RawData, tags, tasks


@tags("tabular_data", "visualization", "correlation")
@tasks("classification", "regression")
def FeatureTargetCorrelationPlot(dataset, fig_height=600) -> Tuple[go.Figure, RawData]:
    """
    Visualizes the correlation between input features and the model's target output in a color-coded horizontal bar
    plot.

    ### Purpose

    This test is designed to graphically illustrate the correlations between distinct input features and the target
    output of a Machine Learning model. Understanding how each feature influences the model's predictions is crucial—a
    higher correlation indicates a stronger influence of the feature on the target variable. This correlation study is
    especially advantageous during feature selection and for comprehending the model's operation.

    ### Test Mechanism

    This FeatureTargetCorrelationPlot test computes and presents the correlations between the features and the target
    variable using a specific dataset. These correlations are calculated and are then graphically represented in a
    horizontal bar plot, color-coded based on the strength of the correlation. A hovering template can also be utilized
    for informative tooltips. It is possible to specify the features to be analyzed and adjust the graph's height
    according to need.

    ### Signs of High Risk

    - There are no strong correlations (either positive or negative) between features and the target variable. This
    could suggest high risk as the supplied features do not appear to significantly impact the prediction output.
    - The presence of duplicated correlation values might hint at redundancy in the feature set.

    ### Strengths

    - Provides visual assistance to interpreting correlations more effectively.
    - Gives a clear and simple tour of how each feature affects the model's target variable.
    - Beneficial for feature selection and grasping the model's prediction nature.
    - Precise correlation values for each feature are offered by the hover template, contributing to a granular-level
    comprehension.

    ### Limitations

    - The test only accepts numerical data, meaning variables of other types need to be prepared beforehand.
    - The plot assumes all correlations to be linear, thus non-linear relationships might not be captured effectively.
    - Not apt for models that employ complex feature interactions, like Decision Trees or Neural Networks, as the test
    may not accurately reflect their importance.
    """
    df = dataset.df[dataset.feature_columns + [dataset.target_column]]

    fig, correlations = _visualize_feature_target_correlation(
        df, dataset.target_column, fig_height
    )

    return fig, RawData(correlation_data=correlations, dataset=dataset.input_id)


def _visualize_feature_target_correlation(df, target_column, fig_height):
    # Compute correlations with the target variable
    correlations = (
        df.corr(numeric_only=True)[target_column].drop(target_column).to_frame()
    )
    correlations = correlations.loc[:, ~correlations.columns.duplicated()]

    correlations = correlations.sort_values(by=target_column, ascending=True)

    # Create a gradual color map from red (1) to blue (-1)
    color_map = np.linspace(1, -1, len(correlations))
    colors = [
        f"rgb({int(255 * (1 - val))}, 0, {int(255 * (1 + val))})" for val in color_map
    ]

    # Create a horizontal bar plot with gradual color mapping
    fig = go.Figure(
        data=go.Bar(
            x=correlations[target_column],
            y=correlations.index,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="Feature: %{y}<br>Correlation: %{x:.2f}<extra></extra>",  # Hover template for tooltips
        )
    )

    # Set the title and axis labels
    fig.update_layout(
        title=f"Correlations of Features vs Target Variable ({target_column})",
        xaxis_title="",
        yaxis_title="",
        height=fig_height,  # Adjust the height value as needed
    )

    return fig, correlations
