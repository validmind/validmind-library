# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from validmind import RawData, tags, tasks
from validmind.logging import get_logger
from validmind.utils import infer_datatypes
from validmind.vm_models import VMDataset

DEFAULT_HISTOGRAM_BINS = 10
DEFAULT_HISTOGRAM_BIN_SIZES = [5, 10, 20, 50]

logger = get_logger(__name__)


def get_numerical_histograms(df, column):
    """
    Returns a collection of histograms for a numerical column, each one
    with a different bin size
    """
    values = df[column].to_numpy()
    values_cleaned = values[~np.isnan(values)]

    # bins='sturges'. Cannot use 'auto' until we review and fix its performance
    #  on datasets with too many unique values
    #
    # 'sturges': R's default method, only accounts for data size. Only optimal
    # for gaussian data and underestimates number of bins for large non-gaussian datasets.
    default_hist = np.histogram(values_cleaned, bins="sturges")

    histograms = {
        "default": {
            "bin_size": len(default_hist[0]),
            "histogram": {
                "bin_edges": default_hist[1].tolist(),
                "counts": default_hist[0].tolist(),
            },
        }
    }

    for bin_size in DEFAULT_HISTOGRAM_BIN_SIZES:
        hist = np.histogram(values_cleaned, bins=bin_size)
        histograms[f"bins_{bin_size}"] = {
            "bin_size": bin_size,
            "histogram": {
                "bin_edges": hist[1].tolist(),
                "counts": hist[0].tolist(),
            },
        }

    return histograms


def get_column_histograms(df, column, type_):
    """
    Returns a collection of histograms for a numerical or categorical column.
    We store different combinations of bin sizes to allow analyzing the data better

    Will be used in favor of _get_histogram in the future
    """
    # Set the minimum number of bins to nunique if it's less than the default
    if type_ == "Numeric":
        return get_numerical_histograms(df, column)
    elif type_ == "Categorical" or type_ == "Boolean":
        value_counts = df[column].value_counts()
        return {
            "default": {
                "bin_size": len(value_counts),
                "histogram": value_counts.to_dict(),
            }
        }
    elif type_ == "Text":
        # Combine all the text in the specified column
        text_data = " ".join(df[column].astype(str))
        # Split the text into words (tokens) using a regular expression
        words = re.findall(r"\w+", text_data)
        # Use Counter to count the frequency of each word
        word_counts = Counter(words)

        return {
            "default": {
                "bin_size": len(word_counts),
                "histogram": dict(word_counts),
            }
        }
    elif type_ == "Null":
        logger.info(f"Ignoring histogram generation for null column {column}")
    else:
        raise ValueError(
            f"Unsupported column type found when computing its histogram: {type_}"
        )


def describe_column(df, column):
    """
    Gets descriptive statistics for a single column in a Pandas DataFrame.
    """
    column_type = column["type"]

    # Initialize statistics with count for all column types
    column["statistics"] = {
        "count": df[column["id"]].count(),
        "n_missing": df[column["id"]].isna().sum(),
        "missing": df[column["id"]].isna().sum() / len(df[column["id"]]),
        "n_distinct": df[column["id"]].nunique(),
        "distinct": df[column["id"]].nunique() / len(df[column["id"]]),
    }

    # Boolean (binary) columns should be reported as categorical
    if column_type == "Boolean" or df[column["id"]].nunique() == 2:
        column["type"] = "Categorical"  # Change the type to Categorical
        top_value = df[column["id"]].value_counts().nlargest(1)
        column["statistics"].update(
            {
                "unique": df[column["id"]].nunique(),
                "top": top_value.index[0],
                "freq": top_value.values[0],
            }
        )
    elif column_type == "Numeric":
        column["statistics"].update(
            df[column["id"]]
            .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
            .to_dict()
        )
    elif column_type == "Categorical" or column_type == "Text":
        column["statistics"].update(
            df[column["id"]].astype("category").describe().to_dict()
        )

    column["histograms"] = get_column_histograms(
        df, column["id"], column["type"]
    )  # Use updated type

    return column


@tags("tabular_data", "time_series_data", "text_data")
@tasks("classification", "regression", "text_classification", "text_summarization")
def DatasetDescription(
    dataset: VMDataset,
) -> Tuple[Dict[str, List[Dict[str, Any]]], RawData]:
    """
    Provides comprehensive analysis and statistical summaries of each column in a machine learning model's dataset.

    ### Purpose

    The test depicted in the script is meant to run a comprehensive analysis on a Machine Learning model's datasets.
    The test or metric is implemented to obtain a complete summary of the columns in the dataset, including vital
    statistics of each column such as count, distinct values, missing values, histograms for numerical, categorical,
    boolean, and text columns. This summary gives a comprehensive overview of the dataset to better understand the
    characteristics of the data that the model is trained on or evaluates.

    ### Test Mechanism

    The DatasetDescription class accomplishes the purpose as follows: firstly, the test method "run" infers the data
    type of each column in the dataset and stores the details (id, column type). For each column, the
    "describe_column" method is invoked to collect statistical information about the column, including count,
    missing value count and its proportion to the total, unique value count, and its proportion to the total. Depending
    on the data type of a column, histograms are generated that reflect the distribution of data within the column.
    Numerical columns use the "get_numerical_histograms" method to calculate histogram distribution, whereas for
    categorical, boolean and text columns, a histogram is computed with frequencies of each unique value in the
    datasets. For unsupported types, an error is raised. Lastly, a summary table is built to aggregate all the
    statistical insights and histograms of the columns in a dataset.

    ### Signs of High Risk

    - High ratio of missing values to total values in one or more columns which may impact the quality of the
    predictions.
    - Unsupported data types in dataset columns.
    - Large number of unique values in the dataset's columns which might make it harder for the model to establish
    patterns.
    - Extreme skewness or irregular distribution of data as reflected in the histograms.

    ### Strengths

    - Provides a detailed analysis of the dataset with versatile summaries like count, unique values, histograms, etc.
    - Flexibility in handling different types of data: numerical, categorical, boolean, and text.
    - Useful in detecting problems in the dataset like missing values, unsupported data types, irregular data
    distribution, etc.
    - The summary gives a comprehensive understanding of dataset features allowing developers to make informed
    decisions.

    ### Limitations

    - The computation can be expensive from a resource standpoint, particularly for large datasets with numerous columns.
    - The histograms use an arbitrary number of bins which may not be the optimal number of bins for specific data
    distribution.
    - Unsupported data types for columns will raise an error which may limit evaluating the dataset.
    - Columns with all null or missing values are not included in histogram computation.
    - This test only validates the quality of the dataset but doesn't address the model's performance directly.
    """
    df = dataset.df

    results = []
    for column in infer_datatypes(df):
        results.append(describe_column(df, column))

    raw_data = {
        column["id"]: {
            "type": column["type"],
            "statistics": column["statistics"],
            "histograms": column["histograms"],
        }
        for column in results
    }

    return {
        "Dataset Description": [
            {
                "Name": column["id"],
                "Type": column["type"],
                "Count": column["statistics"]["count"],
                "Missing": column["statistics"]["n_missing"],
                "Missing %": column["statistics"]["missing"],
                "Distinct": column["statistics"]["n_distinct"],
                "Distinct %": column["statistics"]["distinct"],
            }
            for column in results
        ]
    }, RawData(raw_data=raw_data, dataset=dataset.input_id)
