# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from validmind import tags, tasks
from validmind.logging import get_logger

logger = get_logger(__name__)


@tags("data_validation", "visualization", "time_series_data")
@tasks("regression", "time_series_forecasting")
def TimeSeriesHistogram(dataset, nbins=30) -> Tuple[go.Figure]:
    """
    Visualizes distribution of time-series data using histograms and Kernel Density Estimation (KDE) lines.

    ### Purpose

    The TimeSeriesHistogram test aims to perform a histogram analysis on time-series data to assess the distribution of
    values within a dataset over time. This test is useful for regression tasks and can be applied to various types of
    data, such as internet traffic, stock prices, and weather data, providing insights into the probability
    distribution, skewness, and kurtosis of the dataset.

    ### Test Mechanism

    This test operates on a specific column within the dataset that must have a datetime type index. For each column in
    the dataset, a histogram is created using Plotly's histplot function. If the dataset includes more than one
    time-series, a distinct histogram is plotted for each series. Additionally, a Kernel Density Estimate (KDE) line is
    drawn for each histogram, visualizing the data's underlying probability distribution. The x and y-axis labels are
    hidden to focus solely on the data distribution.

    ### Signs of High Risk

    - The dataset lacks a column with a datetime type index.
    - The specified columns do not exist within the dataset.
    - High skewness or kurtosis in the data distribution, indicating potential bias.
    - Presence of significant outliers in the data distribution.

    ### Strengths

    - Serves as a visual diagnostic tool for understanding data behavior and distribution trends.
    - Effective for analyzing both single and multiple time-series data.
    - KDE line provides a smooth estimate of the overall trend in data distribution.

    ### Limitations

    - Provides a high-level view without specific numeric measures such as skewness or kurtosis.
    - The histogram loses some detail due to binning of data values.
    - Cannot handle non-numeric data columns.
    - Histogram shape may be sensitive to the number of bins used.
    """

    df = dataset.df

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError(f"Dataset {dataset.input_id} must have a datetime index")

    columns = list(dataset.df.columns)

    if not set(columns).issubset(set(df.columns)):
        raise ValueError("Provided 'columns' must exist in the dataset")

    figures = []
    for col in columns:
        # Check for missing values and log if any are found
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            logger.info(
                f"Column '{col}' contains {missing_count} missing values which will be excluded from the histogram."
            )

        # Drop missing values for the current column
        valid_data = df[~df[col].isna()]

        fig = px.histogram(
            valid_data,
            x=col,
            marginal="violin",
            nbins=nbins,
            title=f"Histogram for {col}",
        )
        fig.update_layout(
            title={
                "text": f"{col} (n={len(valid_data)})",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="",
            yaxis_title="",
            font=dict(size=18),
        )
        figures.append(fig)

    return tuple(figures)
