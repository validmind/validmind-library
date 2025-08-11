# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset, VMModel

try:
    from scipy.stats import kstest
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for ModelPredictionResiduals. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e


@tags("regression")
@tasks("residual_analysis", "visualization")
def ModelPredictionResiduals(
    dataset: VMDataset,
    model: VMModel,
    nbins: int = 100,
    p_value_threshold: float = 0.05,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Assesses normality and behavior of residuals in regression models through visualization and statistical tests.

    ### Purpose

    The Model Prediction Residuals test aims to visualize the residuals of model predictions and assess their normality
    using the Kolmogorov-Smirnov (KS) test. It helps to identify potential issues related to model assumptions and
    effectiveness.

    ### Test Mechanism

    The function calculates residuals and generates
    two figures: one for the time series of residuals and one for the histogram of residuals.
    It also calculates the KS test for normality and summarizes the results in a table.

    ### Signs of High Risk

    - Residuals are not normally distributed, indicating potential issues with model assumptions.
    - High skewness or kurtosis in the residuals, which may suggest model misspecification.

    ### Strengths

    - Provides clear visualizations of residuals over time and their distribution.
    - Includes statistical tests to assess the normality of residuals.
    - Helps in identifying potential model misspecifications and assumption violations.

    ### Limitations

    - Assumes that the dataset is provided as a DataFrameDataset object with a .df attribute to access the pandas
    DataFrame.
    - Only generates plots for datasets with a datetime index, resulting in errors for other types of indices.
    """

    df = dataset.df.copy()

    # Filter DataFrame by date range if specified
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    y_true = dataset.y
    y_pred = dataset.y_pred(model)
    residuals = y_true - y_pred

    figures = []

    # Plot residuals
    residuals_fig = go.Figure()
    residuals_fig.add_trace(
        go.Scatter(x=df.index, y=residuals, mode="markers", name="Residuals")
    )
    residuals_fig.update_layout(
        title="Residuals",
        yaxis_title="Residuals",
        font=dict(size=16),
        showlegend=False,
    )
    figures.append(residuals_fig)

    # Plot histogram of residuals
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=residuals, nbinsx=nbins, name="Residuals"))
    hist_fig.update_layout(
        title="Histogram of Residuals",
        xaxis_title="Residuals",
        yaxis_title="Frequency",
        font=dict(size=16),
        showlegend=False,
    )
    figures.append(hist_fig)

    # Perform KS normality test
    ks_stat, p_value = kstest(
        residuals, "norm", args=(residuals.mean(), residuals.std())
    )
    ks_normality = "Normal" if p_value > p_value_threshold else "Not Normal"

    summary = {
        "KS Statistic": ks_stat,
        "p-value": p_value,
        "KS Normality": ks_normality,
        "p-value Threshold": p_value_threshold,
    }

    # Create a summary DataFrame for the KS normality test results
    summary_df = pd.DataFrame([summary])

    return (
        summary_df,
        *figures,
        RawData(residuals=residuals, model=model.input_id, dataset=dataset.input_id),
    )
