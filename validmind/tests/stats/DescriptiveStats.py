# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.utils import format_records
from validmind.vm_models import VMDataset


def _validate_columns(dataset: VMDataset, columns: Optional[List[str]]):
    """Validate and return numerical columns (excluding boolean columns)."""
    if columns is None:
        # Get all columns marked as numeric
        numeric_columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        numeric_columns = [col for col in columns if col in available_columns]

    # Filter out boolean columns as they can't have proper statistical measures computed
    columns = []
    for col in numeric_columns:
        dtype = dataset.df[col].dtype
        # Only include integer and float types, exclude boolean
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            columns.append(col)

    if not columns:
        raise SkipTestError(
            "No numerical columns (integer/float) found for descriptive statistics"
        )

    return columns


def _compute_basic_stats(column: str, data, total_count: int):
    """Compute basic statistics for a column."""
    return {
        "Feature": column,
        "Count": len(data),
        "Missing": total_count - len(data),
        "Missing %": ((total_count - len(data)) / total_count) * 100,
        "Mean": data.mean(),
        "Median": data.median(),
        "Std": data.std(),
        "Min": data.min(),
        "Max": data.max(),
        "Q1": data.quantile(0.25),
        "Q3": data.quantile(0.75),
        "IQR": data.quantile(0.75) - data.quantile(0.25),
    }


def _compute_advanced_stats(column: str, data, confidence_level: float):
    """Compute advanced statistics for a column."""
    try:
        # Distribution measures
        skewness = stats.skew(data)
        kurtosis_val = stats.kurtosis(data)
        cv = (data.std() / data.mean()) * 100 if data.mean() != 0 else np.nan

        # Confidence interval for mean
        ci_lower, ci_upper = stats.t.interval(
            confidence_level,
            len(data) - 1,
            loc=data.mean(),
            scale=data.std() / np.sqrt(len(data)),
        )

        # Normality test
        if len(data) <= 5000:
            normality_stat, normality_p = stats.shapiro(data)
            normality_test = "Shapiro-Wilk"
        else:
            ad_result = stats.anderson(data, dist="norm")
            normality_stat = ad_result.statistic
            normality_p = 0.05 if normality_stat > ad_result.critical_values[2] else 0.1
            normality_test = "Anderson-Darling"

        # Outlier detection using IQR method
        iqr = data.quantile(0.75) - data.quantile(0.25)
        lower_bound = data.quantile(0.25) - 1.5 * iqr
        upper_bound = data.quantile(0.75) + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(data)) * 100

        return {
            "Feature": column,
            "Skewness": skewness,
            "Kurtosis": kurtosis_val,
            "CV %": cv,
            f"CI Lower ({confidence_level * 100:.0f}%)": ci_lower,
            f"CI Upper ({confidence_level * 100:.0f}%)": ci_upper,
            "Normality Test": normality_test,
            "Normality Stat": normality_stat,
            "Normality p-value": normality_p,
            "Normal Distribution": "Yes" if normality_p > 0.05 else "No",
            "Outliers (IQR)": outlier_count,
            "Outliers %": outlier_pct,
        }
    except Exception:
        return None


@tags("tabular_data", "statistics", "data_quality")
@tasks("classification", "regression", "clustering")
def DescriptiveStats(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    include_advanced: bool = True,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Provides comprehensive descriptive statistics for numerical features in a dataset.

    ### Purpose

    This test generates detailed descriptive statistics for numerical features, including
    basic statistics, distribution measures, confidence intervals, and normality tests.
    It provides a comprehensive overview of data characteristics essential for
    understanding data quality and distribution properties.

    ### Test Mechanism

    The test computes various statistical measures for each numerical column:
    - Basic statistics: count, mean, median, std, min, max, quartiles
    - Distribution measures: skewness, kurtosis, coefficient of variation
    - Confidence intervals for the mean
    - Normality tests (Shapiro-Wilk for small samples, Anderson-Darling for larger)
    - Missing value analysis

    ### Signs of High Risk

    - High skewness or kurtosis indicating non-normal distributions
    - Large coefficients of variation suggesting high data variability
    - Significant results in normality tests when normality is expected
    - High percentage of missing values
    - Extreme outliers based on IQR analysis

    ### Strengths

    - Comprehensive statistical analysis in a single test
    - Includes advanced statistical measures beyond basic descriptives
    - Provides confidence intervals for uncertainty quantification
    - Handles missing values appropriately
    - Suitable for both exploratory and confirmatory analysis

    ### Limitations

    - Limited to numerical features only
    - Normality tests may not be meaningful for all data types
    - Large datasets may make some tests computationally expensive
    - Interpretation requires statistical knowledge
    """
    # Validate inputs
    columns = _validate_columns(dataset, columns)

    # Compute statistics
    basic_stats = []
    advanced_stats = []

    for column in columns:
        data = dataset.df[column].dropna()
        total_count = len(dataset.df[column])

        if len(data) == 0:
            continue

        # Basic statistics
        basic_row = _compute_basic_stats(column, data, total_count)
        basic_stats.append(basic_row)

        # Advanced statistics
        if include_advanced and len(data) > 2:
            advanced_row = _compute_advanced_stats(column, data, confidence_level)
            if advanced_row is not None:
                advanced_stats.append(advanced_row)

    # Format results
    results = {}
    if basic_stats:
        results["Basic Statistics"] = format_records(pd.DataFrame(basic_stats))

    if advanced_stats and include_advanced:
        results["Advanced Statistics"] = format_records(pd.DataFrame(advanced_stats))

    if not results:
        raise SkipTestError("Unable to compute statistics for any columns")

    return results
