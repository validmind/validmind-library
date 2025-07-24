# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("time_series_data", "visualization")
@tasks("classification", "regression")
def TabularDateTimeHistograms(dataset: VMDataset) -> Tuple[go.Figure, RawData]:
    """
    Generates histograms to provide graphical insight into the distribution of time intervals in a model's datetime
    data.

    ### Purpose

    The `TabularDateTimeHistograms` metric is designed to provide graphical insight into the distribution of time
    intervals in a machine learning model's datetime data. By plotting histograms of differences between consecutive
    date entries in all datetime variables, it enables an examination of the underlying pattern of time series data and
    identification of anomalies.

    ### Test Mechanism

    This test operates by first identifying all datetime columns and extracting them from the dataset. For each
    datetime column, it next computes the differences (in days) between consecutive dates, excluding zero values, and
    visualizes these differences in a histogram. The Plotly library's histogram function is used to generate
    histograms, which are labeled appropriately and provide a graphical representation of the frequency of different
    day intervals in the dataset.

    ### Signs of High Risk

    - If no datetime columns are detected in the dataset, this would lead to a ValueError. Hence, the absence of
    datetime columns signifies a high risk.
    - A severely skewed or irregular distribution depicted in the histogram may indicate possible complications with
    the data, such as faulty timestamps or abnormalities.

    ### Strengths

    - The metric offers a visual overview of time interval frequencies within the dataset, supporting the recognition
    of inherent patterns.
    - Histogram plots can aid in the detection of potential outliers and data anomalies, contributing to an assessment
    of data quality.
    - The metric is versatile, compatible with a range of task types, including classification and regression, and can
    work with multiple datetime variables if present.

    ### Limitations

    - A major weakness of this metric is its dependence on the visual examination of data, as it does not provide a
    measurable evaluation of the model.
    - The metric might overlook complex or multi-dimensional trends in the data.
    - The test is only applicable to datasets containing datetime columns and will fail if such columns are unavailable.
    - The interpretation of the histograms relies heavily on the domain expertise and experience of the reviewer.
    """
    df = dataset.df
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise SkipTestError("Index must be a datetime type")

    date_diffs = df.index.to_series().sort_values().diff().dt.days.dropna()
    date_diffs = date_diffs[date_diffs != 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=date_diffs, nbinsx=30))
    fig.update_layout(
        title="Index",
        xaxis_title="Days Between Consecutive Dates",
        yaxis_title="Frequency",
        font=dict(size=18),
    )

    return fig, RawData(date_differences=date_diffs, dataset=dataset.input_id)
