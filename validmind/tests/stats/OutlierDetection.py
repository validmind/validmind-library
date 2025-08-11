# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.utils import format_records
from validmind.vm_models import VMDataset


def _validate_columns(dataset: VMDataset, columns: Optional[List[str]]):
    """Validate and return numerical columns."""
    if columns is None:
        columns = dataset.feature_columns_numeric
    else:
        available_columns = set(dataset.feature_columns_numeric)
        columns = [col for col in columns if col in available_columns]

    # Filter out boolean columns as they can't be used for outlier detection
    numeric_columns = []
    for col in columns:
        if col in dataset.df.columns:
            col_dtype = dataset.df[col].dtype
            # Exclude boolean and object types, keep only true numeric types
            if pd.api.types.is_numeric_dtype(col_dtype) and col_dtype != bool:
                numeric_columns.append(col)

    columns = numeric_columns

    if not columns:
        raise SkipTestError("No suitable numerical columns found for outlier detection")

    return columns


def _detect_iqr_outliers(data, iqr_threshold: float):
    """Detect outliers using IQR method."""
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_threshold * iqr
    upper_bound = q3 + iqr_threshold * iqr
    # Fix numpy boolean operation error by using pandas boolean indexing properly
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    iqr_outliers = data[outlier_mask]
    return len(iqr_outliers), (len(iqr_outliers) / len(data)) * 100


def _detect_zscore_outliers(data, zscore_threshold: float):
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data))
    # Fix potential numpy boolean operation error
    outlier_mask = z_scores > zscore_threshold
    zscore_outliers = data[outlier_mask]
    return len(zscore_outliers), (len(zscore_outliers) / len(data)) * 100


def _detect_isolation_forest_outliers(data, contamination: float):
    """Detect outliers using Isolation Forest method."""
    if len(data) <= 10:
        return 0, 0

    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(data.values.reshape(-1, 1))
        iso_outliers = data[outlier_pred == -1]
        return len(iso_outliers), (len(iso_outliers) / len(data)) * 100
    except Exception:
        return 0, 0


def _process_column_outliers(
    column: str,
    data,
    methods: List[str],
    iqr_threshold: float,
    zscore_threshold: float,
    contamination: float,
):
    """Process outlier detection for a single column."""
    outliers_dict = {"Feature": column, "Total Count": len(data)}

    # IQR method
    if "iqr" in methods:
        count, percentage = _detect_iqr_outliers(data, iqr_threshold)
        outliers_dict["IQR Outliers"] = count
        outliers_dict["IQR %"] = percentage

    # Z-score method
    if "zscore" in methods:
        count, percentage = _detect_zscore_outliers(data, zscore_threshold)
        outliers_dict["Z-Score Outliers"] = count
        outliers_dict["Z-Score %"] = percentage

    # Isolation Forest method
    if "isolation_forest" in methods:
        count, percentage = _detect_isolation_forest_outliers(data, contamination)
        outliers_dict["Isolation Forest Outliers"] = count
        outliers_dict["Isolation Forest %"] = percentage

    return outliers_dict


@tags("tabular_data", "statistics", "outliers")
@tasks("classification", "regression", "clustering")
def OutlierDetection(
    dataset: VMDataset,
    columns: Optional[List[str]] = None,
    methods: List[str] = ["iqr", "zscore", "isolation_forest"],
    iqr_threshold: float = 1.5,
    zscore_threshold: float = 3.0,
    contamination: float = 0.1,
) -> Dict[str, Any]:
    """
    Detects outliers in numerical features using multiple statistical methods.

    ### Purpose

    This test identifies outliers in numerical features using various statistical
    methods including IQR, Z-score, and Isolation Forest. It provides comprehensive
    outlier detection to help identify data quality issues and potential anomalies.

    ### Test Mechanism

    The test applies multiple outlier detection methods:
    - IQR method: Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
    - Z-score method: Values with |z-score| > threshold
    - Isolation Forest: ML-based anomaly detection

    ### Signs of High Risk

    - High percentage of outliers indicating data quality issues
    - Inconsistent outlier detection across methods
    - Extreme outliers that significantly deviate from normal patterns

    ### Strengths

    - Multiple detection methods for robust outlier identification
    - Customizable thresholds for different sensitivity levels
    - Clear summary of outlier patterns across features

    ### Limitations

    - Limited to numerical features only
    - Some methods assume normal distributions
    - Threshold selection can be subjective
    """
    # Validate inputs
    columns = _validate_columns(dataset, columns)

    # Process each column
    outlier_summary = []
    for column in columns:
        data = dataset._df[column].dropna()

        if len(data) >= 3:
            outliers_dict = _process_column_outliers(
                column, data, methods, iqr_threshold, zscore_threshold, contamination
            )
            outlier_summary.append(outliers_dict)

    # Format results
    results = {}
    if outlier_summary:
        results["Outlier Summary"] = format_records(pd.DataFrame(outlier_summary))

    return results
