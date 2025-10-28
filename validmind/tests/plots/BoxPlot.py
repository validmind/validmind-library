# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


def _validate_inputs(
    dataset: VMDataset, columns: Optional[List[str]], group_by: Optional[str]
):
    """Validate inputs and return validated columns."""

    # Get dtypes without loading data into memory
    if not isinstance(columns, list):
        columns = [columns]

    columns_dtypes = dataset._df[columns].dtypes

    columns_numeric = []
    columns_numeric = columns_dtypes[
        columns_dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))
    ].index.tolist()

    if columns is None:
        columns = columns_numeric
    else:
        available_columns = set(columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for box plotting")

    if group_by is not None:
        if group_by not in dataset._df.columns:
            raise SkipTestError(f"Group column '{group_by}' not found in dataset")
        if group_by in columns:
            columns.remove(group_by)

    return columns


def _create_grouped_boxplot(
    dataset, columns, group_by, colors, show_outliers, title_prefix, width, height
):
    """Create grouped box plots."""
    fig = go.Figure()
    groups = dataset.df[group_by].dropna().unique()

    for col_idx, column in enumerate(columns):
        for group_idx, group_value in enumerate(groups):
            data_subset = dataset.df[dataset.df[group_by] == group_value][
                column
            ].dropna()

            if len(data_subset) > 0:
                color = colors[group_idx % len(colors)]
                fig.add_trace(
                    go.Box(
                        y=data_subset,
                        name=f"{group_value}",
                        marker_color=color,
                        boxpoints="outliers" if show_outliers else False,
                        jitter=0.3,
                        pointpos=-1.8,
                        legendgroup=f"{group_value}",
                        showlegend=(col_idx == 0),
                        offsetgroup=group_idx,
                        x=[column] * len(data_subset),
                    )
                )

    fig.update_layout(
        title=f"{title_prefix} Features by {group_by}",
        xaxis_title="Features",
        yaxis_title="Values",
        boxmode="group",
        width=width,
        height=height,
        template="plotly_white",
    )
    return fig


def _create_single_boxplot(
    dataset, column, colors, show_outliers, title_prefix, width, height
):
    """Create single column box plot."""
    data = dataset._df[column].dropna()
    if len(data) == 0:
        raise SkipTestError(f"No data available for column {column}")

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=data,
            name=column,
            marker_color=colors[0],
            boxpoints="outliers" if show_outliers else False,
            jitter=0.3,
            pointpos=-1.8,
        )
    )

    fig.update_layout(
        title=f"{title_prefix} {column}",
        yaxis_title=column,
        width=width,
        height=height,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def _create_multiple_boxplots(
    dataset, columns, colors, show_outliers, title_prefix, width, height
):
    """Create multiple column box plots in subplot layout."""
    n_cols = min(2, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    subplot_titles = [f"{title_prefix} {col}" for col in columns]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.2,  # Increased vertical spacing between plots
        horizontal_spacing=0.15,  # Increased horizontal spacing between plots
    )

    for idx, column in enumerate(columns):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        data = dataset._df[column].dropna()

        if len(data) > 0:
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Box(
                    y=data,
                    name=column,
                    marker_color=color,
                    boxpoints="outliers" if show_outliers else False,
                    jitter=0.3,
                    pointpos=-1.8,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(title_text=column, row=row, col=col)
        else:
            fig.add_annotation(
                text=f"No data available<br>for {column}",
                x=0.5,
                y=0.5,
                xref=f"x{idx + 1} domain" if idx > 0 else "x domain",
                yref=f"y{idx + 1} domain" if idx > 0 else "y domain",
                showarrow=False,
                row=row,
                col=col,
            )

    fig.update_layout(
        title="Dataset Feature Distributions",
        width=width,
        height=height,
        template="plotly_white",
        showlegend=False,
    )
    return fig


@tags("tabular_data", "visualization", "data_quality")
@tasks("classification", "regression", "clustering")
def BoxPlot(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    width: int = 1800,
    height: int = 1200,
    colors: Optional[List[str]] = None,
    show_outliers: bool = True,
    title_prefix: str = "Box Plot of",
) -> go.Figure:
    """
    Generates customizable box plots for numerical features in a dataset with optional grouping using Plotly.

    ### Purpose

    This test provides a flexible way to visualize the distribution of numerical features
    through interactive box plots, with optional grouping by categorical variables. Box plots are
    effective for identifying outliers, comparing distributions across groups, and
    understanding the spread and central tendency of the data.

    ### Test Mechanism

    The test creates interactive box plots for specified numerical columns (or all numerical columns
    if none specified). It supports various customization options including:
    - Grouping by categorical variables
    - Customizable colors and styling
    - Outlier display options
    - Interactive hover information
    - Zoom and pan capabilities

    ### Signs of High Risk

    - Presence of many outliers indicating data quality issues
    - Highly skewed distributions
    - Large differences in variance across groups
    - Unexpected patterns in grouped data

    ### Strengths

    - Clear visualization of distribution statistics (median, quartiles, outliers)
    - Interactive Plotly plots with hover information and zoom capabilities
    - Effective for comparing distributions across groups
    - Handles missing values appropriately
    - Highly customizable appearance

    ### Limitations

    - Limited to numerical features only
    - May not be suitable for continuous variables with many unique values
    - Visual interpretation may be subjective
    - Less effective with very large datasets
    """
    # Validate inputs
    columns = _validate_inputs(dataset, columns, group_by)

    # Set default colors
    if colors is None:
        colors = [
            "steelblue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

    # Create appropriate plot type
    if group_by is not None:
        return _create_grouped_boxplot(
            dataset,
            columns,
            group_by,
            colors,
            show_outliers,
            title_prefix,
            width,
            height,
        )
    elif len(columns) == 1:
        return _create_single_boxplot(
            dataset, columns[0], colors, show_outliers, title_prefix, width, height
        )
    else:
        return _create_multiple_boxplots(
            dataset, columns, colors, show_outliers, title_prefix, width, height
        )
