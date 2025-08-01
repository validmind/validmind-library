# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd

from validmind import tags, tasks
from validmind.errors import MissingDependencyError

try:
    from scipy.stats import kurtosis, skew
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for TimeSeriesDescriptiveStatistics. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e


@tags("time_series_data", "analysis")
@tasks("regression")
def TimeSeriesDescriptiveStatistics(dataset) -> pd.DataFrame:
    """
    Evaluates the descriptive statistics of a time series dataset to identify trends, patterns, and data quality issues.

    ### Purpose

    The purpose of the TimeSeriesDescriptiveStatistics function is to analyze an individual time series by providing a
    summary of key descriptive statistics. This analysis helps in understanding trends, patterns, and data quality
    issues within the time series dataset.

    ### Test Mechanism

    The function extracts the time series data and provides a summary of key descriptive statistics. The dataset is
    expected to have a datetime index, and the function will check this and raise an error if the index is not in a
    datetime format. For each variable (column) in the dataset, appropriate statistics, including start date, end date,
    min, mean, max, skewness, kurtosis, and count, are calculated.

    ### Signs of High Risk

    - If the index of the dataset is not in datetime format, it could lead to errors in time-series analysis.
    - Inconsistent or missing data within the dataset might affect the analysis of trends and patterns.

    ### Strengths

    - Provides a comprehensive summary of key descriptive statistics for each variable.
    - Helps identify data quality issues and understand the distribution of the data.

    ### Limitations

    - Assumes the dataset is provided as a DataFrameDataset object with a .df attribute to access the pandas DataFrame.
    - Only analyzes datasets with a datetime index and will raise an error for other types of indices.
    - Does not handle large datasets efficiently, and performance may degrade with very large datasets.
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
        count = df[column].count()
        min_value = df[column].min()
        mean_value = df[column].mean()
        max_value = df[column].max()
        skewness_value = skew(df[column].dropna())
        kurtosis_value = kurtosis(df[column].dropna())

        summary.append(
            {
                "Variable": column,
                "Start Date": start_date,
                "End Date": end_date,
                "Min": min_value,
                "Mean": mean_value,
                "Max": max_value,
                "Skewness": skewness_value,
                "Kurtosis": kurtosis_value,
                "Count": count,
            }
        )

    result_df = pd.DataFrame(summary)

    return result_df
