# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional

import plotly.express as px

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("tabular_data", "visualization", "distribution")
@tasks("classification", "regression", "clustering")
def ViolinPlot(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    width: int = 800,
    height: int = 600,
) -> px.violin:
    """
    Generates interactive violin plots for numerical features using Plotly.

    ### Purpose

    This test creates violin plots to visualize the distribution of numerical features,
    showing both the probability density and summary statistics. Violin plots combine
    aspects of box plots and kernel density estimation for rich distribution visualization.

    ### Test Mechanism

    The test creates violin plots for specified numerical columns, with optional
    grouping by categorical variables. Each violin shows the distribution shape,
    quartiles, and median values.

    ### Signs of High Risk

    - Multimodal distributions that might indicate mixed populations
    - Highly skewed distributions suggesting data quality issues
    - Large differences in distribution shapes across groups
    - Unusual distribution patterns that contradict domain expectations

    ### Strengths

    - Shows detailed distribution shape information
    - Interactive Plotly visualization with hover details
    - Effective for comparing distributions across groups
    - Combines density estimation with quartile information

    ### Limitations

    - Limited to numerical features only
    - Requires sufficient data points for meaningful density estimation
    - May not be suitable for discrete variables
    - Can be misleading with very small sample sizes
    """
    # Get numerical columns
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for violin plot")

    # For violin plots, we'll melt the data to long format
    data = dataset.df[columns].dropna()

    if len(data) == 0:
        raise SkipTestError("No valid data available for violin plot")

    # Melt the dataframe to long format
    melted_data = data.melt(var_name="Feature", value_name="Value")

    # Add group column if specified
    if group_by and group_by in dataset.df.columns:
        # Repeat group values for each feature
        group_values = []
        for column in columns:
            column_data = dataset.df[[column, group_by]].dropna()
            group_values.extend(column_data[group_by].tolist())

        if len(group_values) == len(melted_data):
            melted_data["Group"] = group_values
        else:
            group_by = None  # Disable grouping if lengths don't match

    # Create violin plot
    if group_by and "Group" in melted_data.columns:
        fig = px.violin(
            melted_data,
            x="Feature",
            y="Value",
            color="Group",
            box=True,
            title=f"Distribution of Features by {group_by}",
            width=width,
            height=height,
        )
    else:
        fig = px.violin(
            melted_data,
            x="Feature",
            y="Value",
            box=True,
            title="Feature Distributions",
            width=width,
            height=height,
        )

    # Update layout
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Features",
        yaxis_title="Values",
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    return fig
