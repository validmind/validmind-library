# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("time_series_data")
@tasks("regression")
def TimeSeriesOutliers(
    dataset: VMDataset, zscore_threshold: int = 3
) -> Tuple[pd.DataFrame, List[go.Figure], bool, RawData]:
    """
    Identifies and visualizes outliers in time-series data using the z-score method.

    ### Purpose

    This test is designed to identify outliers in time-series data using the z-score method. It's vital for ensuring
    data quality before modeling, as outliers can skew predictive models and significantly impact their overall
    performance.

    ### Test Mechanism

    The test processes a given dataset which must have datetime indexing, checks if a 'zscore_threshold' parameter has
    been supplied, and identifies columns with numeric data types. After finding numeric columns, the implementer then
    applies the z-score method to each numeric column, identifying outliers based on the threshold provided. Each
    outlier is listed together with their variable name, z-score, timestamp, and relative threshold in a dictionary and
    converted to a DataFrame for convenient output. Additionally, it produces visual plots for each time series
    illustrating outliers in the context of the broader dataset. The 'zscore_threshold' parameter sets the limit beyond
    which a data point will be labeled as an outlier. The default threshold is set at 3, indicating that any data point
    that falls 3 standard deviations away from the mean will be marked as an outlier.

    ### Signs of High Risk

    - Many or substantial outliers are present within the dataset, indicating significant anomalies.
    - Data points with z-scores higher than the set threshold.
    - Potential impact on the performance of machine learning models if outliers are not properly addressed.

    ### Strengths

    - The z-score method is a popular and robust method for identifying outliers in a dataset.
    - Simplifies time series maintenance by requiring a datetime index.
    - Identifies outliers for each numeric feature individually.
    - Provides an elaborate report showing variables, dates, z-scores, and pass/fail tests.
    - Offers visual inspection for detected outliers through plots.

    ### Limitations

    - The test only identifies outliers in numeric columns, not in categorical variables.
    - The utility and accuracy of z-scores can be limited if the data doesn't follow a normal distribution.
    - The method relies on a subjective z-score threshold for deciding what constitutes an outlier, which might not
    always be suitable depending on the dataset and use case.
    - It does not address possible ways to handle identified outliers in the data.
    - The requirement for a datetime index could limit its application.
    """
    df = dataset.df

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise SkipTestError("Dataset must be provided with datetime index")

    df_numeric = df[dataset.feature_columns_numeric]
    z_scores = pd.DataFrame(
        data=df_numeric.apply(lambda x: (x - x.mean()) / x.std()),
        index=df.index,
        columns=dataset.feature_columns_numeric,
    )

    outlier_table = []
    outliers = z_scores[(z_scores.abs() > zscore_threshold).any(axis=1)]

    for idx, row in outliers.iterrows():
        for col in dataset.feature_columns_numeric:
            if abs(row[col]) > zscore_threshold:
                outlier_table.append(
                    {
                        "Column": col,
                        "Z-Score": row[col],
                        "Threshold": zscore_threshold,
                        "Date": idx.strftime("%Y-%m-%d"),
                        "Pass/Fail": "Fail",
                    }
                )

    outlier_df = pd.DataFrame(outlier_table)
    figures = []

    for column in outlier_df["Column"].unique():
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=df.index, y=df[column], mode="lines", name="Time Series")
        )

        column_outliers = outlier_df[outlier_df["Column"] == column]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(column_outliers["Date"]),
                y=df.loc[pd.to_datetime(column_outliers["Date"]), column],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Outliers",
            )
        )

        fig.update_layout(
            title=f"Outliers for {column}", xaxis_title="Date", yaxis_title=column
        )

        figures.append(fig)

    return (
        outlier_df.sort_values(["Column", "Date"]),
        figures,
        len(outlier_df) == 0,
        RawData(outliers=outlier_df, dataset=dataset.input_id),
    )
