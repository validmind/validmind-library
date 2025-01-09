# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import plotly.graph_objects as go

from validmind import tags, tasks


@tags("model_predictions", "visualization")
@tasks("regression", "time_series_forecasting")
def TimeSeriesPredictionsPlot(dataset, model):
    """
    Plot actual vs predicted values for time series data and generate a visual comparison for the model.

    ### Purpose

    The purpose of this function is to visualize the actual versus predicted values for time
    series data for a single model.

    ### Test Mechanism

    The function plots the actual values from the dataset and overlays the predicted
    values from the model using Plotly for interactive visualization.

    - Large discrepancies between actual and predicted values indicate poor model performance.
    - Systematic deviations in predicted values can highlight model bias or issues with data patterns.

    ### Strengths

    - Provides a clear visual comparison of model predictions against actual values.
    - Uses Plotly for interactive and visually appealing plots.

    ### Limitations

    - Assumes that the dataset is provided as a DataFrameDataset object with a datetime index.
    - Requires that `dataset.y_pred(model)` returns the predicted values for the model.
    """
    fig = go.Figure()

    # Plot actual values from the dataset
    time_index = dataset.df.index  # Assuming the index of the dataset is datetime
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=dataset.y,
            mode="lines",
            name="Actual",
            line=dict(color="blue"),
        )
    )

    # Plot predicted values for the model
    y_pred = dataset.y_pred(model)
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="orange"),  # Using a distinct color for the prediction
        )
    )

    # Update layout
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Time",
        yaxis_title="Values",
        template="plotly_white",
    )

    return fig
