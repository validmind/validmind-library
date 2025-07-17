# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd

from validmind import tags, tasks


@tags("time_series_data", "analysis")
@tasks("regression")
def TimeSeriesDescription(dataset) -> pd.DataFrame:
    """
    Generates a detailed analysis for the provided time series dataset, summarizing key statistics to identify trends,
    patterns, and data quality issues.

    ### Purpose

    The TimeSeriesDescription function aims to analyze an individual time series by providing a summary of key
    statistics. This helps in understanding trends, patterns, and data quality issues within the time series.

    ### Test Mechanism

    The function extracts the time series data and provides a summary of key statistics. The dataset is expected to
    have a datetime index. The function checks this and raises an error if the index is not in datetime format. For
    each variable (column) in the dataset, appropriate statistics including start date, end date, frequency, number of
    missing values, count, min, and max values are calculated.

    ### Signs of High Risk

    - If the index of the dataset is not in datetime format, it could lead to errors in time-series analysis.
    - Inconsistent or missing data within the dataset might affect the analysis of trends and patterns.

    ### Strengths

    - Provides a comprehensive summary of key statistics for each variable, helping to identify data quality issues
    such as missing values.
    - Helps in understanding the distribution and range of the data by including min and max values.

    ### Limitations

    - Assumes that the dataset is provided as a DataFrameDataset object with a .df attribute to access the pandas
    DataFrame.
    - Only analyzes datasets with a datetime index and will raise an error for other types of indices.
    - Does not handle large datasets efficiently; performance may degrade with very large datasets.
    """

    summary = []

    df = (
        dataset.df
    )  # Assuming DataFrameDataset objects have a .df attribute to get the pandas DataFrame

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError(f"Dataset {dataset.input_id} must have a datetime index")

    for column in df.columns:
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")
        frequency = pd.infer_freq(df.index)
        num_missing_values = df[column].isna().sum()
        count = df[column].count()
        min_value = df[column].min()
        max_value = df[column].max()

        summary.append(
            {
                "Variable": column,
                "Start Date": start_date,
                "End Date": end_date,
                "Frequency": frequency,
                "Num of Missing Values": num_missing_values,
                "Count": count,
                "Min Value": min_value,
                "Max Value": max_value,
            }
        )

    result_df = pd.DataFrame(summary)

    return result_df
