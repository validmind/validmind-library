# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset


@tags("tabular_data", "data_quality")
@tasks("classification", "regression")
def MissingValues(
    dataset: VMDataset,
    min_percentage_threshold: float = 1.0,
) -> Tuple[List[Dict[str, Any]], bool, RawData]:
    """
    Evaluates dataset quality by ensuring missing value percentage across all features does not exceed a set threshold.

    ### Purpose

    The Missing Values test is designed to evaluate the quality of a dataset by measuring the number of missing values
    across all features. The objective is to ensure that the ratio of missing data to total data is less than a
    predefined threshold (as a percentage), defaulting to 1.0, in order to maintain the data quality necessary for
    reliable predictive strength in a machine learning model.

    ### Test Mechanism

    The mechanism for this test involves iterating through each column of the dataset, counting missing values
    (represented as NaNs), and calculating the percentage they represent against the total number of rows. The test
    then checks if the missing value percentage is less than or equal to the predefined `min_percentage_threshold`. The results are
    shown in a table summarizing each column, the number of missing values, the percentage of missing values in each
    column, and a Pass/Fail status based on the threshold comparison.

    ### Signs of High Risk

    - When the missing value percentage in any column exceeds the `min_percentage_threshold` value.
    - Presence of missing values across many columns, leading to multiple instances of failing the threshold.

    ### Strengths

    - Quick and granular identification of missing data across each feature in the dataset.
    - Provides an effective and straightforward means of maintaining data quality, essential for constructing efficient
    machine learning models.

    ### Limitations

    - Does not suggest the root causes of the missing values or recommend ways to impute or handle them.
    - May overlook features with significant missing data but still less than the `min_percentage_threshold`, potentially
    impacting the model.
    - Does not account for data encoded as values like "-999" or "None," which might not technically classify as
    missing but could bear similar implications.
    """
    df = dataset.df
    missing = df.isna().sum()
    n_rows = df.shape[0]
    missing_pct = (missing / n_rows * 100) if n_rows else (missing * 0.0)

    return (
        [
            {
                "Column": col,
                "Number of Missing Values": missing[col],
                "Percentage of Missing Values (%)": missing_pct[col],
                "Pass/Fail": "Pass"
                if missing_pct[col] <= min_percentage_threshold
                else "Fail",
            }
            for col in missing.index
        ],
        all(missing_pct[col] <= min_percentage_threshold for col in missing.index),
        RawData(missing_values=missing, dataset=dataset.input_id),
    )
