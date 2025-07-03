# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
from statsmodels.sandbox.stats.runs import runstest_1samp

from validmind import tags, tasks


@tasks("classification", "regression")
@tags("tabular_data", "statistical_test", "statsmodels")
def RunsTest(dataset) -> pd.DataFrame:
    """
    Executes Runs Test on ML model to detect non-random patterns in output data sequence.

    ### Purpose

    The Runs Test is a statistical procedure used to determine whether the sequence of data extracted from the ML model
    behaves randomly or not. Specifically, it analyzes runs, sequences of consecutive positives or negatives, in the
    data to check if there are more or fewer runs than expected under the assumption of randomness. This can be an
    indication of some pattern, trend, or cycle in the model's output which may need attention.

    ### Test Mechanism

    The testing mechanism applies the Runs Test from the statsmodels module on each column of the training dataset. For
    every feature in the dataset, a Runs Test is executed, whose output includes a Runs Statistic and P-value. A low
    P-value suggests that data arrangement in the feature is not likely to be random. The results are stored in a
    dictionary where the keys are the feature names, and the values are another dictionary storing the test statistic
    and the P-value for each feature.

    ### Signs of High Risk

    - High risk is indicated when the P-value is close to zero.
    - If the P-value is less than a predefined significance level (like 0.05), it suggests that the runs (series of
    positive or negative values) in the model's output are not random and are longer or shorter than what is expected
    under a random scenario.
    - This would mean there's a high risk of non-random distribution of errors or model outcomes, suggesting potential
    issues with the model.

    ### Strengths

    - Straightforward and fast for detecting non-random patterns in data sequence.
    - Validates assumptions of randomness, which is valuable for checking error distributions in regression models,
    trendless time series data, and ensuring a classifier doesn't favor one class over another.
    - Can be applied to both classification and regression tasks, making it versatile.

    ### Limitations

    - Assumes that the data is independently and identically distributed (i.i.d.), which might not be the case for many
    real-world datasets.
    - The conclusion drawn from the low P-value indicating non-randomness does not provide information about the type
    or the source of the detected pattern.
    - Sensitive to extreme values (outliers), and overly large or small run sequences can influence the results.
    - Does not provide model performance evaluation; it is used to detect patterns in the sequence of outputs only.
    """

    df = dataset.df[dataset.feature_columns_numeric]

    runs_test_values = {}
    for col in df.columns:
        runs_stat, runs_p_value = runstest_1samp(df[col].values)
        runs_test_values[col] = {
            "stat": runs_stat,
            "pvalue": runs_p_value,
        }

    runs_test_df = pd.DataFrame.from_dict(runs_test_values, orient="index")
    runs_test_df.reset_index(inplace=True)
    runs_test_df.columns = ["feature", "stat", "pvalue"]

    return runs_test_df
