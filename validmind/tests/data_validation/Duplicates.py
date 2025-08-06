# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, Tuple

import pandas as pd

from validmind import tags, tasks


@tags("tabular_data", "data_quality", "text_data")
@tasks("classification", "regression")
def Duplicates(dataset, min_threshold=1) -> Tuple[Dict[str, Any], bool]:
    """
    Tests dataset for duplicate entries, ensuring model reliability via data quality verification.

    ### Purpose

    The 'Duplicates' test is designed to check for duplicate rows within the dataset provided to the model. It serves
    as a measure of data quality, ensuring that the model isn't merely memorizing duplicate entries or being swayed by
    redundant information. This is an important step in the pre-processing of data for both classification and
    regression tasks.

    ### Test Mechanism

    This test operates by checking each row for duplicates in the dataset. If a text column is specified in the
    dataset, the test is conducted on this column; if not, the test is run on all feature columns. The number and
    percentage of duplicates are calculated and returned in a DataFrame. Additionally, a test is passed if the total
    count of duplicates falls below a specified minimum threshold.

    ### Signs of High Risk

    - A high number of duplicate rows in the dataset, which can lead to overfitting where the model performs well on
    the training data but poorly on unseen data.
    - A high percentage of duplicate rows in the dataset, indicating potential problems with data collection or
    processing.

    ### Strengths

    - Assists in improving the reliability of the model's training process by ensuring the training data is not
    contaminated with duplicate entries, which can distort statistical analyses.
    - Provides both absolute numbers and percentage values of duplicate rows, giving a thorough overview of data
    quality.
    - Highly customizable as it allows for setting a user-defined minimum threshold to determine if the test has been
    passed.

    ### Limitations

    - Does not distinguish between benign duplicates (i.e., coincidental identical entries in different rows) and
    problematic duplicates originating from data collection or processing errors.
    - The test becomes more computationally intensive as the size of the dataset increases, which might not be suitable
    for very large datasets.
    - Can only check for exact duplicates and may miss semantically similar information packaged differently.
    """
    df = dataset.df[dataset.text_column or dataset.feature_columns]

    duplicate_rows_count = df.duplicated().sum()
    percentage_duplicate_rows = (duplicate_rows_count / len(df)) * 100

    result_df = pd.DataFrame(
        {
            "Number of Duplicates": [duplicate_rows_count],
            "Percentage of Rows (%)": [percentage_duplicate_rows],
        }
    )

    # test has passed if the total sum of duplicates is less than the threshold
    passed = result_df["Number of Duplicates"].sum() < min_threshold

    return {
        "Duplicate Rows Results for Dataset": result_df.to_dict(orient="records")
    }, passed
