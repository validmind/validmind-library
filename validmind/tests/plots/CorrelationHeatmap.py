# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


def _validate_and_prepare_data(
    dataset: VMDataset, columns: Optional[List[str]], method: str
):
    """Validate inputs and prepare correlation data."""
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for correlation analysis")

    if len(columns) < 2:
        raise SkipTestError(
            "At least 2 numerical columns required for correlation analysis"
        )

    # Get data and remove constant columns
    data = dataset.df[columns]
    data = data.loc[:, data.var() != 0]

    if data.shape[1] < 2:
        raise SkipTestError(
            "Insufficient non-constant columns for correlation analysis"
        )

    return data.corr(method=method)


def _apply_filters(corr_matrix, threshold: Optional[float], mask_upper: bool):
    """Apply threshold and masking filters to correlation matrix."""
    if threshold is not None:
        mask = np.abs(corr_matrix) < threshold
        corr_matrix = corr_matrix.mask(mask)

    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix = corr_matrix.mask(mask)

    return corr_matrix


def _create_annotation_text(z_values, y_labels, x_labels, show_values: bool):
    """Create text annotations for heatmap cells."""
    if not show_values:
        return None

    text = []
    for i in range(len(y_labels)):
        text_row = []
        for j in range(len(x_labels)):
            value = z_values[i][j]
            if np.isnan(value):
                text_row.append("")
            else:
                text_row.append(f"{value:.3f}")
        text.append(text_row)
    return text


def _calculate_adaptive_font_size(n_features: int) -> int:
    """Calculate adaptive font size based on number of features."""
    if n_features <= 10:
        return 12
    elif n_features <= 20:
        return 10
    elif n_features <= 30:
        return 8
    else:
        return 6


def _calculate_stats_and_update_layout(
    fig, corr_matrix, method: str, title: str, width: int, height: int
):
    """Calculate statistics and update figure layout."""
    n_features = corr_matrix.shape[0]
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    upper_triangle = upper_triangle[~np.isnan(upper_triangle)]

    if len(upper_triangle) > 0:
        mean_corr = np.abs(upper_triangle).mean()
        max_corr = np.abs(upper_triangle).max()
        stats_text = f"Features: {n_features}<br>Mean |r|: {mean_corr:.3f}<br>Max |r|: {max_corr:.3f}"
    else:
        stats_text = f"Features: {n_features}"

    fig.update_layout(
        title={
            "text": f"{title} ({method.capitalize()} Correlation)",
            "x": 0.5,
            "xanchor": "center",
        },
        width=width,
        height=height,
        template="plotly_white",
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(tickmode="linear", autorange="reversed"),
        annotations=[
            dict(
                text=stats_text,
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
            )
        ],
    )


@tags("tabular_data", "visualization", "correlation")
@tasks("classification", "regression", "clustering")
def CorrelationHeatmap(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    show_values: bool = True,
    colorscale: str = "RdBu",
    width: int = 800,
    height: int = 600,
    mask_upper: bool = False,
    threshold: Optional[float] = None,
    title: str = "Correlation Heatmap",
) -> go.Figure:
    """
    Generates customizable correlation heatmap plots for numerical features in a dataset using Plotly.

    ### Purpose

    This test provides a flexible way to visualize correlations between numerical features
    in a dataset using interactive Plotly heatmaps. It supports different correlation methods
    and extensive customization options for the heatmap appearance, making it suitable for
    exploring feature relationships in data analysis.

    ### Test Mechanism

    The test computes correlation coefficients between specified numerical columns
    (or all numerical columns if none specified) using the specified method.
    It then creates an interactive heatmap visualization with customizable appearance options including:
    - Different correlation methods (pearson, spearman, kendall)
    - Color schemes and annotations
    - Masking options for upper triangle
    - Threshold filtering for significant correlations
    - Interactive hover information

    ### Signs of High Risk

    - Very high correlations (>0.9) between features indicating multicollinearity
    - Unexpected correlation patterns that contradict domain knowledge
    - Features with no correlation to any other variables
    - Strong correlations with the target variable that might indicate data leakage

    ### Strengths

    - Supports multiple correlation methods
    - Interactive Plotly plots with hover information and zoom capabilities
    - Highly customizable visualization options
    - Can handle missing values appropriately
    - Provides clear visual representation of feature relationships
    - Optional thresholding to focus on significant correlations

    ### Limitations

    - Limited to numerical features only
    - Cannot capture non-linear relationships effectively
    - May be difficult to interpret with many features
    - Correlation does not imply causation
    """
    # Validate inputs and compute correlation
    corr_matrix = _validate_and_prepare_data(dataset, columns, method)

    # Apply filters
    corr_matrix = _apply_filters(corr_matrix, threshold, mask_upper)

    # Prepare heatmap data
    z_values = corr_matrix.values
    x_labels = corr_matrix.columns.tolist()
    y_labels = corr_matrix.index.tolist()
    text = _create_annotation_text(z_values, y_labels, x_labels, show_values)

    # Calculate adaptive font size
    n_features = len(x_labels)
    font_size = _calculate_adaptive_font_size(n_features)

    # Create heatmap
    heatmap_kwargs = {
        "z": z_values,
        "x": x_labels,
        "y": y_labels,
        "colorscale": colorscale,
        "zmin": -1,
        "zmax": 1,
        "colorbar": dict(title=f"{method.capitalize()} Correlation"),
        "hoverongaps": False,
        "hovertemplate": "<b>%{y}</b> vs <b>%{x}</b><br>"
        + f"{method.capitalize()} Correlation: %{{z:.3f}}<br>"
        + "<extra></extra>",
    }

    # Add text annotations if requested
    if show_values and text is not None:
        heatmap_kwargs.update(
            {
                "text": text,
                "texttemplate": "%{text}",
                "textfont": {"size": font_size, "color": "black"},
            }
        )

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    # Update layout with stats
    _calculate_stats_and_update_layout(fig, corr_matrix, method, title, width, height)

    return fig
