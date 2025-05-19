# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Any, Dict, Tuple

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset


def compute_outliers(series, threshold=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    return series[(series < lower_bound) | (series > upper_bound)]


@tags("tabular_data", "numerical_data")
@tasks("classification", "regression")
def IQROutliersTable(
    dataset: VMDataset, threshold: float = 1.5
) -> Tuple[Dict[str, Any], RawData]:
    """
    Determines and summarizes outliers in numerical features using the Interquartile Range method.

    ### Purpose

    The "Interquartile Range Outliers Table" (IQROutliersTable) metric is designed to identify and summarize outliers
    within numerical features of a dataset using the Interquartile Range (IQR) method. This exercise is crucial in the
    pre-processing of data because outliers can substantially distort statistical analysis and impact the performance
    of machine learning models.

    ### Test Mechanism

    The IQR, which is the range separating the first quartile (25th percentile) from the third quartile (75th
    percentile), is calculated for each numerical feature within the dataset. An outlier is defined as a data point
    falling below the "Q1 - 1.5 * IQR" or above "Q3 + 1.5 * IQR" range. The test computes the number of outliers and
    their summary statistics (minimum, 25th percentile, median, 75th percentile, and maximum values) for each numerical
    feature. If no specific features are chosen, the test applies to all numerical features in the dataset. The default
    outlier threshold is set to 1.5 but can be customized by the user.

    ### Signs of High Risk

    - A large number of outliers in multiple features.
    - Outliers significantly distanced from the mean value of variables.
    - Extremely high or low outlier values indicative of data entry errors or other data quality issues.

    ### Strengths

    - Provides a comprehensive summary of outliers for each numerical feature, helping pinpoint features with potential
    quality issues.
    - The IQR method is robust to extremely high or low outlier values as it is based on quartile calculations.
    - Can be customized to work on selected features and set thresholds for outliers.

    ### Limitations

    - Might cause false positives if the variable deviates from a normal or near-normal distribution, especially for
    skewed distributions.
    - Does not provide interpretation or recommendations for addressing outliers, relying on further analysis by users
    or data scientists.
    - Only applicable to numerical features, not categorical data.
    - Default thresholds may not be optimal for data with heavy pre-processing, manipulation, or inherently high
    kurtosis (heavy tails).
    """
    df = dataset.df

    outliers_table = []
    all_outliers = {}

    for col in dataset.feature_columns_numeric:
        # Skip binary features
        if len(df[col].unique()) <= 2:
            continue

        outliers = compute_outliers(df[col], threshold)
        all_outliers[col] = outliers

        if outliers.empty:
            continue

        outliers_table.append(
            {
                "Variable": col,
                "Total Count of Outliers": outliers.count(),
                "Mean Value of Variable": df[col].mean(),
                "Minimum Outlier Value": outliers.min(),
                "Outlier Value at 25th Percentile": outliers.quantile(0.25),
                "Outlier Value at 50th Percentile": outliers.median(),
                "Outlier Value at 75th Percentile": outliers.quantile(0.75),
                "Maximum Outlier Value": outliers.max(),
            }
        )

    return {
        "Summary of Outliers Detected by IQR Method": outliers_table,
    }, RawData(all_outliers=all_outliers, dataset=dataset.input_id)
