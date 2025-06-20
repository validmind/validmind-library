# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("time_series_data")
@tasks("regression")
def TimeSeriesFrequency(
    dataset: VMDataset,
) -> Tuple[List[Dict[str, Any]], go.Figure, bool, RawData]:
    """
    Evaluates consistency of time series data frequency and generates a frequency plot.

    ### Purpose

    The purpose of the TimeSeriesFrequency test is to evaluate the consistency in the frequency of data points in a
    time-series dataset. This test inspects the intervals or duration between each data point to determine if a fixed
    pattern (such as daily, weekly, or monthly) exists. The identification of such patterns is crucial to time-series
    analysis as any irregularities could lead to erroneous results and hinder the model's capacity for identifying
    trends and patterns.

    ### Test Mechanism

    Initially, the test checks if the dataframe index is in datetime format. Subsequently, it utilizes pandas'
    `infer_freq` method to identify the frequency of each data series within the dataframe. The `infer_freq` method
    attempts to establish the frequency of a time series and returns both the frequency string and a dictionary
    relating these strings to their respective labels. The test compares the frequencies of all datasets. If they share
    a common frequency, the test passes, but it fails if they do not. Additionally, Plotly is used to create a
    frequency plot, offering a visual depiction of the time differences between consecutive entries in the dataframe
    index.

    ### Signs of High Risk

    - The test fails, indicating multiple unique frequencies within the dataset. This failure could suggest irregular
    intervals between observations, potentially interrupting pattern recognition or trend analysis.
    - The presence of missing or null frequencies could be an indication of inconsistencies in data or gaps within the
    data collection process.

    ### Strengths

    - This test uses a systematic approach to checking the consistency of data frequency within a time-series dataset.
    - It increases the model's reliability by asserting the consistency of observations over time, an essential factor
    in time-series analysis.
    - The test generates a visual plot, providing an intuitive representation of the dataset's frequency distribution,
    which caters to visual learners and aids in interpretation and explanation.

    ### Limitations

    - This test is only applicable to time-series datasets and hence not suitable for other types of datasets.
    - The `infer_freq` method might not always correctly infer frequency when faced with missing or irregular data
    points.
    - Depending on context or the model under development, mixed frequencies might sometimes be acceptable, but this
    test considers them a failing condition.
    """
    df = dataset.df

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise SkipTestError("Dataset must be provided with datetime index")

    frequencies = []
    freq_dict = {
        "S": "Second",
        "T": "Minute",
        "min": "Minute",
        "H": "Hourly",
        "D": "Daily",
        "B": "Business day",
        "W": "Weekly",
        "MS": "Monthly",
        "M": "Monthly",
        "Q": "Quarterly",
        "A": "Yearly",
        "Y": "Yearly",
    }

    for column in dataset.feature_columns_numeric:
        series = df[column].dropna()
        if not series.empty:
            freq = pd.infer_freq(series.index)
            label = freq_dict.get(freq, freq)
        else:
            label = None

        frequencies.append({"Variable": column, "Frequency": label})

    # Calculate the time differences between consecutive entries
    time_diff = df.index.to_series().diff().dropna()
    # Convert the time differences to a suitable unit (e.g., days)
    time_diff_days = time_diff.dt.total_seconds() / (60 * 60 * 24)
    # Plot the time differences as a histogram
    fig = go.Figure(
        data=[go.Histogram(x=time_diff_days, nbinsx=50)],
        layout=go.Layout(
            title="Histogram of Time Differences (Days)",
            xaxis_title="Days",
            yaxis_title="Frequency",
            font=dict(size=16),
        ),
    )

    return (
        frequencies,
        fig,
        len(set(item["Frequency"] for item in frequencies)) == 1,
        RawData(time_diff_days=time_diff_days, dataset=dataset.input_id),
    )
