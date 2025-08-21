# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


def _validate_columns(dataset: VMDataset, columns: Optional[List[str]]):
    """Validate and return numerical columns."""
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for histogram plotting")

    return columns


def _process_column_data(data, log_scale: bool, column: str):
    """Process column data and return plot data and xlabel."""
    plot_data = data
    xlabel = column
    if log_scale and (data > 0).all():
        plot_data = np.log10(data)
        xlabel = f"log10({column})"
    return plot_data, xlabel


def _add_histogram_trace(
    fig, plot_data, bins, color, opacity, normalize, column, row, col
):
    """Add histogram trace to figure."""
    histnorm = "probability density" if normalize else None

    fig.add_trace(
        go.Histogram(
            x=plot_data,
            nbinsx=bins if isinstance(bins, int) else None,
            name=f"Histogram - {column}",
            marker_color=color,
            opacity=opacity,
            histnorm=histnorm,
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _add_kde_trace(fig, plot_data, bins, normalize, column, row, col):
    """Add KDE trace to figure if possible."""
    try:
        kde = stats.gaussian_kde(plot_data)
        x_range = np.linspace(plot_data.min(), plot_data.max(), 100)
        kde_values = kde(x_range)

        if not normalize:
            hist_max = (
                len(plot_data) / bins if isinstance(bins, int) else len(plot_data) / 30
            )
            kde_values = kde_values * hist_max / kde_values.max()

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                mode="lines",
                name=f"KDE - {column}",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    except Exception:
        pass


def _add_stats_annotation(fig, data, idx, row, col):
    """Add statistics annotation to subplot."""
    stats_text = f"Mean: {data.mean():.3f}<br>Std: {data.std():.3f}<br>N: {len(data)}"
    fig.add_annotation(
        text=stats_text,
        x=0.02,
        y=0.98,
        xref=f"x{idx + 1} domain" if idx > 0 else "x domain",
        yref=f"y{idx + 1} domain" if idx > 0 else "y domain",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        row=row,
        col=col,
    )


@tags("tabular_data", "visualization", "data_quality")
@tasks("classification", "regression", "clustering")
def HistogramPlot(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    bins: Union[int, str, List] = 30,
    color: str = "steelblue",
    opacity: float = 0.7,
    show_kde: bool = True,
    normalize: bool = False,
    log_scale: bool = False,
    title_prefix: str = "Histogram of",
    width: int = 1200,
    height: int = 800,
    n_cols: int = 2,
    vertical_spacing: float = 0.15,
    horizontal_spacing: float = 0.1,
) -> go.Figure:
    """
    Generates customizable histogram plots for numerical features in a dataset using Plotly.

    ### Purpose

    This test provides a flexible way to visualize the distribution of numerical features in a dataset.
    It allows for extensive customization of the histogram appearance and behavior through parameters,
    making it suitable for various exploratory data analysis tasks.

    ### Test Mechanism

    The test creates histogram plots for specified numerical columns (or all numerical columns if none specified).
    It supports various customization options including:
    - Number of bins or bin edges
    - Color and opacity
    - Kernel density estimation overlay
    - Logarithmic scaling
    - Normalization options
    - Configurable subplot layout (columns and spacing)

    ### Signs of High Risk

    - Highly skewed distributions that may indicate data quality issues
    - Unexpected bimodal or multimodal distributions
    - Presence of extreme outliers
    - Empty or sparse distributions

    ### Strengths

    - Highly customizable visualization options
    - Interactive Plotly plots with zoom, pan, and hover capabilities
    - Supports both single and multiple column analysis
    - Provides insights into data distribution patterns
    - Can handle different data types and scales
    - Configurable subplot layout for better visualization

    ### Limitations

    - Limited to numerical features only
    - Visual interpretation may be subjective
    - May not be suitable for high-dimensional datasets
    - Performance may degrade with very large datasets
    """
    # Validate inputs
    columns = _validate_columns(dataset, columns)

    # Calculate subplot layout
    n_cols = min(n_cols, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Create subplots
    subplot_titles = [f"{title_prefix} {col}" for col in columns]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    for idx, column in enumerate(columns):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        data = dataset.df[column].dropna()

        if len(data) == 0:
            fig.add_annotation(
                text=f"No data available<br>for {column}",
                x=0.5,
                y=0.5,
                xref=f"x{idx + 1}" if idx > 0 else "x",
                yref=f"y{idx + 1}" if idx > 0 else "y",
                showarrow=False,
                row=row,
                col=col,
            )
            continue

        # Process data
        plot_data, xlabel = _process_column_data(data, log_scale, column)

        # Add histogram
        _add_histogram_trace(
            fig, plot_data, bins, color, opacity, normalize, column, row, col
        )

        # Add KDE if requested
        if show_kde and len(data) > 1:
            _add_kde_trace(fig, plot_data, bins, normalize, column, row, col)

        # Update axes and add annotations
        fig.update_xaxes(title_text=xlabel, row=row, col=col)
        ylabel = "Density" if normalize else "Frequency"
        fig.update_yaxes(title_text=ylabel, row=row, col=col)
        _add_stats_annotation(fig, data, idx, row, col)

    # Update layout
    fig.update_layout(
        title_text="Dataset Feature Distributions",
        showlegend=False,
        width=width,
        height=height,
        template="plotly_white",
    )

    return fig
