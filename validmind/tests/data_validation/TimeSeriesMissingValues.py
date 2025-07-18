# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("time_series_data")
@tasks("regression")
def TimeSeriesMissingValues(
    dataset: VMDataset, min_threshold: int = 1
) -> Tuple[List[Dict[str, Any]], go.Figure, go.Figure, bool, RawData]:
    """
    Validates time-series data quality by confirming the count of missing values is below a certain threshold.

    ### Purpose

    This test is designed to validate the quality of a historical time-series dataset by verifying that the number of
    missing values is below a specified threshold. As time-series models greatly depend on the continuity and
    temporality of data points, missing values could compromise the model's performance. Consequently, this test aims
    to ensure data quality and readiness for the machine learning model, safeguarding its predictive capacity.

    ### Test Mechanism

    The test method commences by validating if the dataset has a datetime index; if not, an error is raised. It
    establishes a lower limit threshold for missing values and performs a missing values check on each column of the
    dataset. An object for the test result is created stating whether the number of missing values is within the
    specified threshold. Additionally, the test calculates the percentage of missing values alongside the raw count.

    ### Signs of High Risk

    - The number of missing values in any column of the dataset surpasses the threshold, marking a failure and a
    high-risk scenario. The reasons could range from incomplete data collection, faulty sensors to data preprocessing
    errors.

    ### Strengths

    - Effectively identifies missing values which could adversely affect the model’s performance.
    - Applicable and customizable through the threshold parameter across different data sets.
    - Goes beyond raw numbers by calculating the percentage of missing values, offering a more relative understanding
    of data scarcity.

    ### Limitations

    - Although it identifies missing values, the test does not provide solutions to handle them.
    - The test demands that the dataset should have a datetime index, hence limiting its use only to time series
    analysis.
    - The test's sensitivity to the 'min_threshold' parameter may raise false alarms if set too strictly or may
    overlook problematic data if set too loosely.
    - Solely focuses on the 'missingness' of the data and might fall short in addressing other aspects of data quality.
    """
    df = dataset.df

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise SkipTestError("Dataset must be provided with datetime index")

    missing = df.isna().sum()

    if sum(missing.values) == 0:
        # if theres no missing values, no need to plot anything
        return (
            [
                {
                    "Column": col,
                    "Number of Missing Values": missing[col],
                    "Percentage of Missing Values (%)": 0,
                    "Pass/Fail": "Pass",
                }
                for col in missing.index
            ],
            True,
        )

    barplot = px.bar(
        missing,
        x=missing.index,
        y=missing.values,
        labels={"x": "", "y": "Missing Values"},
        title="Total Number of Missing Values per Variable",
        color=missing.values,
        color_continuous_scale="Reds",
    )

    missing_mask = df.isnull()
    z = missing_mask.T.astype(int).values
    x = missing_mask.index.tolist()
    y = missing_mask.columns.tolist()
    heatmap = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Reds",
        showscale=False,
    )

    # Update the layout after creation
    heatmap.update_layout(title="Missing Values Heatmap")

    return (
        [
            {
                "Column": col,
                "Number of Missing Values": missing[col],
                "Percentage of Missing Values (%)": missing[col] / df.shape[0] * 100,
                "Pass/Fail": "Pass" if missing[col] < min_threshold else "Fail",
            }
            for col in missing.index
        ],
        barplot,
        heatmap,
        all(missing[col] < min_threshold for col in missing.index),
        RawData(missing_values_count=missing, missing_values_mask=missing_mask),
    )
