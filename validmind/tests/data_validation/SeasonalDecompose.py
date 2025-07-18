# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError, SkipTestError

try:
    from scipy import stats
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for SeasonalDecompose. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags("time_series_data", "seasonality", "statsmodels")
@tasks("regression")
def SeasonalDecompose(
    dataset: VMDataset, seasonal_model: str = "additive"
) -> Tuple[go.Figure, RawData]:
    """
    Assesses patterns and seasonality in a time series dataset by decomposing its features into foundational components.

    ### Purpose

    The Seasonal Decompose test aims to decompose the features of a time series dataset into their fundamental
    components: observed, trend, seasonal, and residuals. By utilizing the Seasonal Decomposition of Time Series by
    Loess (STL) method, the test identifies underlying patterns, predominantly seasonality, in the dataset's features.
    This aids in developing a more comprehensive understanding of the dataset, which in turn facilitates more effective
    model validation.

    ### Test Mechanism

    The testing process leverages the `seasonal_decompose` function from the `statsmodels.tsa.seasonal` library to
    evaluate each feature in the dataset. It isolates each feature into four components—observed, trend, seasonal, and
    residuals—and generates six subplot graphs per feature for visual interpretation. Prior to decomposition, the test
    scrutinizes and removes any non-finite values, ensuring the reliability of the analysis.

    ### Signs of High Risk

    - **Non-Finiteness**: Datasets with a high number of non-finite values may flag as high risk since these values are
    omitted before conducting the seasonal decomposition.
    - **Frequent Warnings**: Chronic failure to infer the frequency for a scrutinized feature indicates high risk.
    - **High Seasonality**: A significant seasonal component could potentially render forecasts unreliable due to
    overwhelming seasonal variation.

    ### Strengths

    - **Seasonality Detection**: Accurately discerns hidden seasonality patterns in dataset features.
    - **Visualization**: Facilitates interpretation and comprehension through graphical representations.
    - **Unrestricted Usage**: Not confined to any specific regression model, promoting wide-ranging applicability.

    ### Limitations

    - **Dependence on Assumptions**: Assumes that dataset features are periodically distributed. Features with no
    inferable frequency are excluded from the test.
    - **Handling Non-Finite Values**: Disregards non-finite values during analysis, potentially resulting in an
    incomplete understanding of the dataset.
    - **Unreliability with Noisy Datasets**: Produces unreliable results when used with datasets that contain heavy
    noise.
    """
    df = dataset.df

    figures = []

    raw_data = {}

    for col in df.columns:
        series = df[col].dropna()

        if series[np.isfinite(series)].empty:
            logger.warning(f"No finite values found for {col}, skipping")
            continue

        inferred_freq = pd.infer_freq(series.index)
        if inferred_freq is None:
            logger.warning(f"No frequency found for {col}, skipping")
            continue

        sd = seasonal_decompose(series[np.isfinite(series)], model=seasonal_model)

        # Create subplots using Plotly
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Observed",
                "Trend",
                "Seasonal",
                "Residuals",
                "Histogram and KDE of Residuals",
                "Normal Q-Q Plot of Residuals",
            ),
            vertical_spacing=0.1,
        )

        # Observed
        fig.add_trace(
            go.Scatter(x=sd.observed.index, y=sd.observed, name="Observed"),
            row=1,
            col=1,
        )
        # Trend
        fig.add_trace(
            go.Scatter(x=sd.trend.index, y=sd.trend, name="Trend"),
            row=1,
            col=2,
        )
        # Seasonal
        fig.add_trace(
            go.Scatter(x=sd.seasonal.index, y=sd.seasonal, name="Seasonal"),
            row=2,
            col=1,
        )
        # Residuals
        fig.add_trace(
            go.Scatter(x=sd.resid.index, y=sd.resid, name="Residuals"),
            row=2,
            col=2,
        )
        # Histogram with KDE
        residuals = sd.resid.dropna()
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=100, name="Residuals"),
            row=3,
            col=1,
        )
        # Normal Q-Q plot
        qq = stats.probplot(residuals, plot=None)
        qq_line_slope, qq_line_intercept = stats.linregress(qq[0][0], qq[0][1])[:2]
        qq_line = qq_line_slope * np.array(qq[0][0]) + qq_line_intercept
        fig.add_trace(
            go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="QQ plot"),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq_line,
                mode="lines",
                name="QQ line",
            ),
            row=3,
            col=2,
        )

        fig.update_layout(
            height=1000,
            title_text=f"Seasonal Decomposition for {col}",
            showlegend=False,
        )

        figures.append(fig)

        # Add the decomposed components to raw_data
        raw_data[col] = {
            "observed": sd.observed,
            "trend": sd.trend,
            "seasonal": sd.seasonal,
            "residuals": sd.resid,
        }

    if not figures:
        raise SkipTestError("No valid features found for seasonal decomposition")

    return (*figures, RawData(decomposed_components=raw_data, dataset=dataset.input_id))
