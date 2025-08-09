# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset, VMModel

try:
    from scipy.stats import norm
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for TimeSeriesPredictionWithCI. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e


@tags("model_predictions", "visualization")
@tasks("regression", "time_series_forecasting")
def TimeSeriesPredictionWithCI(
    dataset: VMDataset,
    model: VMModel,
    confidence: float = 0.95,
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Assesses predictive accuracy and uncertainty in time series models, highlighting breaches beyond confidence
    intervals.

    ### Purpose

    The purpose of the Time Series Prediction with Confidence Intervals (CI) test is to visualize the actual versus
    predicted values for time series data, including confidence intervals, and to compute and report the number of
    breaches beyond these intervals. This helps in evaluating the reliability and accuracy of the model's predictions.

    ### Test Mechanism

    The function performs the following steps:

    - Calculates the standard deviation of prediction errors.
    - Determines the confidence intervals using a specified confidence level, typically 95%.
    - Counts the number of actual values that fall outside the confidence intervals, referred to as breaches.
    - Generates a plot visualizing the actual values, predicted values, and confidence intervals.
    - Returns a DataFrame summarizing the breach information, including the total breaches, upper breaches, and lower
    breaches.

    ### Signs of High Risk

    - A high number of breaches indicates that the model's predictions are not reliable within the specified confidence
    level.
    - Significant deviations between actual and predicted values may highlight model inadequacies or issues with data
    quality.

    ### Strengths

    - Provides a visual representation of prediction accuracy and the uncertainty around predictions.
    - Includes a statistical measure of prediction reliability through confidence intervals.
    - Computes and reports breaches, offering a quantitative assessment of prediction performance.

    ### Limitations

    - Assumes that the dataset is provided as a DataFrameDataset object with a datetime index.
    - Requires that `dataset.y_pred(model)` returns the predicted values for the model.
    - The calculation of confidence intervals assumes normally distributed errors, which may not hold for all datasets.
    """
    time_index = dataset.df.index  # Assuming the index of the dataset is datetime

    # Get actual and predicted values
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Compute the standard deviation of the errors
    errors = y_true - y_pred
    std_error = np.std(errors)

    # Compute z-score for the given confidence level
    z_score = norm.ppf(1 - (1 - confidence) / 2)

    # Compute confidence intervals
    lower_conf = y_pred - z_score * std_error
    upper_conf = y_pred + z_score * std_error

    # Calculate breaches
    upper_breaches = (y_true > upper_conf).sum()
    lower_breaches = (y_true < lower_conf).sum()
    total_breaches = upper_breaches + lower_breaches

    # Create DataFrame
    breaches_df = pd.DataFrame(
        {
            "Confidence Level": [confidence],
            "Total Breaches": [total_breaches],
            "Upper Breaches": [upper_breaches],
            "Lower Breaches": [lower_breaches],
        }
    )

    # Plotting
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="blue"),
        )
    )

    # Plot predicted values
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="red"),
        )
    )

    # Add confidence interval lower bound as an invisible line
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=lower_conf,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="CI Lower",
        )
    )

    # Add confidence interval upper bound and fill area
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=upper_conf,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(200, 200, 200, 0.5)",
            line=dict(width=0),
            showlegend=True,
            name="Confidence Interval",
        )
    )

    # Update layout
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Time",
        yaxis_title="Values",
        template="plotly_white",
    )

    return (
        fig,
        breaches_df,
        RawData(
            errors=errors,
            z_score=z_score,
            lower_confidence=lower_conf,
            upper_confidence=upper_conf,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
