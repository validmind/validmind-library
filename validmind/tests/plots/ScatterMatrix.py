# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional

import plotly.express as px

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("tabular_data", "visualization", "correlation")
@tasks("classification", "regression", "clustering")
def ScatterMatrix(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    color_by: Optional[str] = None,
    max_features: int = 10,
    width: int = 800,
    height: int = 600,
) -> px.scatter_matrix:
    """
    Generates an interactive scatter matrix plot for numerical features using Plotly.

    ### Purpose

    This test creates a scatter matrix visualization to explore pairwise relationships
    between numerical features in a dataset. It provides an efficient way to identify
    correlations, patterns, and outliers across multiple feature combinations.

    ### Test Mechanism

    The test creates a scatter matrix where each cell shows the relationship between
    two features. The diagonal shows the distribution of individual features.
    Optional color coding by categorical variables helps identify group patterns.

    ### Signs of High Risk

    - Strong linear relationships that might indicate multicollinearity
    - Outliers that appear consistently across multiple feature pairs
    - Unexpected clustering patterns in the data
    - No clear relationships between features and target variables

    ### Strengths

    - Interactive Plotly visualization with zoom and hover capabilities
    - Efficient visualization of multiple feature relationships
    - Optional grouping by categorical variables
    - Automatic handling of large feature sets through sampling

    ### Limitations

    - Limited to numerical features only
    - Can become cluttered with too many features
    - Requires sufficient data points for meaningful patterns
    - May not capture non-linear relationships effectively
    """
    # Get numerical columns
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        # Validate columns exist and are numeric
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    if not columns:
        raise SkipTestError("No numerical columns found for scatter matrix")

    # Limit number of features to avoid overcrowding
    if len(columns) > max_features:
        columns = columns[:max_features]

    # Prepare data
    data = dataset.df[columns].dropna()

    if len(data) == 0:
        raise SkipTestError("No valid data available for scatter matrix")

    # Add color column if specified
    if color_by and color_by in dataset.df.columns:
        data = dataset.df[columns + [color_by]].dropna()
        if len(data) == 0:
            raise SkipTestError(f"No valid data available with color column {color_by}")

    # Create scatter matrix
    fig = px.scatter_matrix(
        data,
        dimensions=columns,
        color=color_by if color_by and color_by in data.columns else None,
        title=f"Scatter Matrix for {len(columns)} Features",
        width=width,
        height=height,
    )

    # Update layout
    fig.update_layout(template="plotly_white", title_x=0.5)

    return fig
