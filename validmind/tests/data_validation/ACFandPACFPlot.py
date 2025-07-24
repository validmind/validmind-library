# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset


@tags("time_series_data", "forecasting", "statistical_test", "visualization")
@tasks("regression")
def ACFandPACFPlot(dataset: VMDataset) -> Tuple[go.Figure, RawData]:
    """
    Analyzes time series data using Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to
    reveal trends and correlations.

    ### Purpose

    The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plot test is employed to analyze
    time series data in machine learning models. It illuminates the correlation of the data over time by plotting the
    correlation of the series with its own lags (ACF), and the correlations after removing effects already accounted
    for by earlier lags (PACF). This information can identify trends, such as seasonality, degrees of autocorrelation,
    and inform the selection of order parameters for AutoRegressive Integrated Moving Average (ARIMA) models.

    ### Test Mechanism

    The `ACFandPACFPlot` test accepts a dataset with a time-based index. It first confirms the index is of a datetime
    type, then handles any NaN values. The test subsequently generates ACF and PACF plots for each column in the
    dataset, producing a subplot for each. If the dataset doesn't include key columns, an error is returned.

    ### Signs of High Risk

    - Sudden drops in the correlation at a specific lag might signal a model at high risk.
    - Consistent high correlation across multiple lags could also indicate non-stationarity in the data, which may
    suggest that a model estimated on this data won't generalize well to future, unknown data.

    ### Strengths

    - ACF and PACF plots offer clear graphical representations of the correlations in time series data.
    - These plots are effective at revealing important data characteristics such as seasonality, trends, and
    correlation patterns.
    - The insights from these plots aid in better model configuration, particularly in the selection of ARIMA model
    parameters.

    ### Limitations

    - ACF and PACF plots are exclusively for time series data and hence, can't be applied to all ML models.
    - These plots require large, consistent datasets as gaps could lead to misleading results.
    - The plots can only represent linear correlations and fail to capture any non-linear relationships within the data.
    - The plots might be difficult for non-experts to interpret and should not replace more advanced analyses.
    """
    if not pd.api.types.is_datetime64_any_dtype(dataset.df.index):
        raise ValueError("Index must be a datetime type")

    columns = list(dataset.df.columns)
    df = dataset.df.dropna()

    if not set(columns).issubset(set(df.columns)):
        raise ValueError("Provided 'columns' must exist in the dataset")

    figures = []
    acf_store = {}
    pacf_store = {}
    for col in df.columns:
        series = df[col]
        max_lags = min(40, len(series) // 2 - 1)

        # Create ACF plot using Plotly
        acf_values = acf(series, nlags=max_lags)
        acf_fig = go.Figure()
        acf_fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values))
        acf_fig.update_layout(
            title=f"ACF for {col}",
            xaxis_title="Lag",
            yaxis_title="ACF",
            font=dict(size=18),
        )
        figures.append(acf_fig)
        acf_store[col] = acf_values

        # Create PACF plot using Plotly
        pacf_values = pacf(series, nlags=max_lags)
        pacf_fig = go.Figure()
        pacf_fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values))
        pacf_fig.update_layout(
            title=f"PACF for {col}",
            xaxis_title="Lag",
            yaxis_title="PACF",
            font=dict(size=18),
        )
        figures.append(pacf_fig)
        pacf_store[col] = pacf_values

    return (
        *figures,
        RawData(acf_values=acf_store, pacf_values=pacf_store, dataset=dataset.input_id),
    )
