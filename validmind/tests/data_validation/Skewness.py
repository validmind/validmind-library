# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Tuple

from validmind import tags, tasks
from validmind.utils import infer_datatypes


@tags("data_quality", "tabular_data")
@tasks("classification", "regression")
def Skewness(dataset, max_threshold=1) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
    """
    Evaluates the skewness of numerical data in a dataset to check against a defined threshold, aiming to ensure data
    quality and optimize model performance.

    ### Purpose

    The purpose of the Skewness test is to measure the asymmetry in the distribution of data within a predictive
    machine learning model. Specifically, it evaluates the divergence of said distribution from a normal distribution.
    Understanding the level of skewness helps identify data quality issues, which are crucial for optimizing the
    performance of traditional machine learning models in both classification and regression settings.

    ### Test Mechanism

    This test calculates the skewness of numerical columns in the dataset, focusing specifically on numerical data
    types. The calculated skewness value is then compared against a predetermined maximum threshold, which is set by
    default to 1. If the skewness value is less than this maximum threshold, the test passes; otherwise, it fails. The
    test results, along with the skewness values and column names, are then recorded for further analysis.

    ### Signs of High Risk

    - Substantial skewness levels that significantly exceed the maximum threshold.
    - Persistent skewness in the data, indicating potential issues with the foundational assumptions of the machine
    learning model.
    - Subpar model performance, erroneous predictions, or biased inferences due to skewed data distributions.

    ### Strengths

    - Fast and efficient identification of unequal data distributions within a machine learning model.
    - Adjustable maximum threshold parameter, allowing for customization based on user needs.
    - Provides a clear quantitative measure to mitigate model risks related to data skewness.

    ### Limitations

    - Only evaluates numeric columns, potentially missing skewness or bias in non-numeric data.
    - Assumes that data should follow a normal distribution, which may not always be applicable to real-world data.
    - Subjective threshold for risk grading, requiring expert input and recurrent iterations for refinement.
    """

    # Use the imported infer_datatypes function
    dataset_types = infer_datatypes(dataset.df)

    # Convert the list of dictionaries to a dictionary for easy access
    dataset_types_dict = {item["id"]: item["type"] for item in dataset_types}

    skewness = dataset.df.skew(numeric_only=True)

    results_table = []
    passed = True

    for col in skewness.index:
        if dataset_types_dict.get(col) != "Numeric":
            continue

        col_skewness = skewness[col]
        col_passed = abs(col_skewness) < max_threshold
        passed = passed and col_passed

        results_table.append(
            {
                "Column": col,
                "Skewness": col_skewness,
                "Pass/Fail": "Pass" if col_passed else "Fail",
            }
        )

    return (
        {
            "Skewness Results for Dataset": results_table,
        },
        passed,
    )
