# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


import pandas as pd
import plotly.express as px
from sklearn import metrics

from validmind import tags, tasks


@tags("model_performance", "sklearn")
@tasks("regression", "time_series_forecasting")
def TimeSeriesR2SquareBySegments(dataset, model, segments=None):
    """
    Evaluates the R-Squared values of regression models over specified time segments in time series data to assess
    segment-wise model performance.

    ### Purpose

    The TimeSeriesR2SquareBySegments test aims to evaluate the R-Squared values for several regression models across
    different segments of time series data. This helps in determining how well the models explain the variability in
    the data within each specific time segment.

    ### Test Mechanism
    - Provides a visual representation of model performance across different time segments.
    - Allows for identification of segments where the model performs poorly.
    - Calculating the R-Squared values for each segment.
    - Generating a bar chart to visually represent the R-Squared values across different models and segments.

    ### Signs of High Risk

    - Significantly low R-Squared values for certain time segments, indicating poor model performance in those periods.
    - Large variability in R-Squared values across different segments for the same model, suggesting inconsistent
    performance.

    ### Strengths

    - Provides a visual representation of how well models perform over different time periods.
    - Helps identify time segments where models may need improvement or retraining.
    - Facilitates comparison between multiple models in a straightforward manner.

    ### Limitations

    - Assumes datasets are provided as DataFrameDataset objects with the attributes `y`, `y_pred`, and
    `feature_columns`.
    - Requires that `dataset.y_pred(model)` returns predicted values for the model.
    - Assumes that both `y_true` and `y_pred` are pandas Series with datetime indices, which may not always be the case.
    - May not account for more nuanced temporal dependencies within the segments.
    """
    results_list = []

    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Ensure y_true and y_pred are pandas Series with the same index
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=dataset.df.index)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=dataset.df.index)

    index = dataset.df.index

    if segments is None:
        mid_point = len(index) // 2
        segments = {
            "start_date": [index.min(), index[mid_point]],
            "end_date": [index[mid_point - 1], index.max()],
        }

    for segment_index, (start_date, end_date) in enumerate(
        zip(segments["start_date"], segments["end_date"])
    ):
        mask = (index >= start_date) & (index <= end_date)
        y_true_segment = y_true.loc[mask]
        y_pred_segment = y_pred.loc[mask]

        if len(y_true_segment) > 0 and len(y_pred_segment) > 0:
            r2s = metrics.r2_score(y_true_segment, y_pred_segment)
            results_list.append(
                {
                    "Segments": f"Segment {segment_index + 1}",
                    "Start Date": start_date,
                    "End Date": end_date,
                    "R-Squared": r2s,
                }
            )

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Plotting
    fig = px.bar(
        results_df,
        x="Segments",
        y="R-Squared",
        # color="Model",
        barmode="group",
        title="R-Squared by Segment",
        labels={
            "R-Squared": "R-Squared Value",
            "Segments": "Time Segment",
            # "Model": "Model",
        },
    )

    return fig, results_df
