# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Any, Dict, List, Tuple

from validmind.tests import tags, tasks
from validmind.vm_models import VMDataset


@tags("tabular_data")
@tasks("regression", "classification")
def TooManyZeroValues(
    dataset: VMDataset, max_percent_threshold: float = 0.03
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Identifies numerical columns in a dataset that contain an excessive number of zero values, defined by a threshold
    percentage.

    ### Purpose

    The 'TooManyZeroValues' test is utilized to identify numerical columns in the dataset that may present a quantity
    of zero values considered excessive. The aim is to detect situations where these may implicate data sparsity or a
    lack of variation, limiting their effectiveness within a machine learning model. The definition of 'too many' is
    quantified as a percentage of total values, with a default set to 0.03%.

    ### Test Mechanism

    This test is conducted by looping through each column in the dataset and categorizing those that pertain to
    numerical data. On identifying a numerical column, the function computes the total quantity of zero values and
    their ratio to the total row count. Should the proportion exceed a pre-set threshold parameter, set by default at
    0.03%, the column is considered to have failed the test. The results for each column are summarized and
    reported, indicating the count and percentage of zero values for each numerical column, alongside a status
    indicating whether the column has passed or failed the test.

    ### Signs of High Risk

    - Numerical columns showing a high ratio of zero values when compared to the total count of rows (exceeding the
    predetermined threshold).
    - Columns characterized by zero values across the board suggest a complete lack of data variation, signifying high
    risk.

    ### Strengths

    - Assists in highlighting columns featuring an excess of zero values that could otherwise go unnoticed within a
    large dataset.
    - Provides the flexibility to alter the threshold that determines when the quantity of zero values becomes 'too
    many', thus catering to specific needs of a particular analysis or model.
    - Offers feedback in the form of both counts and percentages of zero values, which allows a closer inspection of
    the distribution and proportion of zeros within a column.
    - Targets specifically numerical data, thereby avoiding inappropriate application to non-numerical columns and
    mitigating the risk of false test failures.

    ### Limitations

    - Is exclusively designed to check for zero values and doesn’t assess the potential impact of other values that
    could affect the dataset, such as extremely high or low figures, missing values, or outliers.
    - Lacks the ability to detect a repetitive pattern of zeros, which could be significant in time-series or
    longitudinal data.
    - Zero values can actually be meaningful in some contexts; therefore, tagging them as 'too many' could potentially
    misinterpret the data to some extent.
    - This test does not take into consideration the context of the dataset, and fails to recognize that within certain
    columns, a high number of zero values could be quite normal and not necessarily an indicator of poor data quality.
    - Cannot evaluate non-numerical or categorical columns, which might bring with them different types of concerns or
    issues.
    """
    df = dataset.df
    table = []

    for col in dataset.feature_columns_numeric:
        value_counts = df[col].value_counts()
        row_count = df.shape[0]

        if 0 not in value_counts.index:
            continue

        n_zeros = value_counts[0]
        p_zeros = (n_zeros / row_count) * 100

        table.append(
            {
                "Variable": col,
                "Row Count": row_count,
                "Number of Zero Values": n_zeros,
                "Percentage of Zero Values (%)": p_zeros,
                "Pass/Fail": ("Pass" if p_zeros < (max_percent_threshold) else "Fail"),
            }
        )

    return table, all(row["Pass/Fail"] == "Pass" for row in table)
