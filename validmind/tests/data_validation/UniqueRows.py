# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

from validmind import tags, tasks
from validmind.vm_models import VMDataset


@tags("tabular_data")
@tasks("regression", "classification")
def UniqueRows(
    dataset: VMDataset, min_percent_threshold: float = 1
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Verifies the diversity of the dataset by ensuring that the count of unique rows exceeds a prescribed threshold.

    ### Purpose

    The UniqueRows test is designed to gauge the quality of the data supplied to the machine learning model by
    verifying that the count of distinct rows in the dataset exceeds a specific threshold, thereby ensuring a varied
    collection of data. Diversity in data is essential for training an unbiased and robust model that excels when faced
    with novel data.

    ### Test Mechanism

    The testing process starts with calculating the total number of rows in the dataset. Subsequently, the count of
    unique rows is determined for each column in the dataset. If the percentage of unique rows (calculated as the ratio
    of unique rows to the overall row count) is less than the prescribed minimum percentage threshold given as a
    function parameter, the test passes. The results are cached and a final pass or fail verdict is given based on
    whether all columns have successfully passed the test.

    ### Signs of High Risk

    - A lack of diversity in data columns, demonstrated by a count of unique rows that falls short of the preset
    minimum percentage threshold, is indicative of high risk.
    - This lack of variety in the data signals potential issues with data quality, possibly leading to overfitting in
    the model and issues with generalization, thus posing a significant risk.

    ### Strengths

    - The UniqueRows test is efficient in evaluating the data's diversity across each information column in the dataset.
    - This test provides a quick, systematic method to assess data quality based on uniqueness, which can be pivotal in
    developing effective and unbiased machine learning models.

    ### Limitations

    - A limitation of the UniqueRows test is its assumption that the data's quality is directly proportionate to its
    uniqueness, which may not always hold true. There might be contexts where certain non-unique rows are essential and
    should not be overlooked.
    - The test does not consider the relative 'importance' of each column in predicting the output, treating all
    columns equally.
    - This test may not be suitable or useful for categorical variables, where the count of unique categories is
    inherently limited.
    """
    df = dataset.df

    rows = df.shape[0]
    unique_rows = df.nunique()

    table = [
        {
            "Column": col,
            "Number of Unique Values": unique_rows[col],
            "Percentage of Unique Values (%)": unique_rows[col] / rows * 100,
            "Pass/Fail": (
                "Pass"
                if (unique_rows[col] / rows * 100) >= min_percent_threshold
                else "Fail"
            ),
        }
        for col in unique_rows.index
    ]

    return table, all(row["Pass/Fail"] == "Pass" for row in table)
