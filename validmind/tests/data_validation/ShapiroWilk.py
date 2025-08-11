# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import pandas as pd

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError

try:
    from scipy import stats
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for ShapiroWilk. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e


@tasks("classification", "regression")
@tags("tabular_data", "data_distribution", "statistical_test")
def ShapiroWilk(dataset) -> Tuple[pd.DataFrame, RawData]:
    """
    Evaluates feature-wise normality of training data using the Shapiro-Wilk test.

    ### Purpose

    The Shapiro-Wilk test is utilized to investigate whether a particular dataset conforms to the standard normal
    distribution. This analysis is crucial in machine learning modeling because the normality of the data can
    profoundly impact the performance of the model. This metric is especially useful in evaluating various features of
    the dataset in both classification and regression tasks.

    ### Test Mechanism

    The Shapiro-Wilk test is conducted on each feature column of the training dataset to determine if the data
    contained fall within the normal distribution. The test presents a statistic and a p-value, with the p-value
    serving to validate or repudiate the null hypothesis, which is that the tested data is normally distributed.

    ### Signs of High Risk

    - A p-value that falls below 0.05 signifies a high risk as it discards the null hypothesis, indicating that the
    data does not adhere to the normal distribution.
    - For machine learning models built on the presumption of data normality, such an outcome could result in subpar
    performance or incorrect predictions.

    ### Strengths

    - The Shapiro-Wilk test is esteemed for its level of accuracy, thereby making it particularly well-suited to
    datasets of small to moderate sizes.
    - It proves its versatility through its efficient functioning in both classification and regression tasks.
    - By separately testing each feature column, the Shapiro-Wilk test can raise an alarm if a specific feature does
    not comply with the normality.

    ### Limitations

    - The Shapiro-Wilk test's sensitivity can be a disadvantage as it often rejects the null hypothesis (i.e., data is
    normally distributed), even for minor deviations, especially in large datasets. This may lead to unwarranted 'false
    alarms' of high risk by deeming the data as not normally distributed even if it approximates normal distribution.
    - Exceptional care must be taken in managing missing data or outliers prior to testing as these can greatly skew
    the results.
    - Lastly, the Shapiro-Wilk test is not optimally suited for processing data with pronounced skewness or kurtosis.
    """

    df = dataset.df[dataset.feature_columns_numeric]

    sw_values = {}
    for col in df.columns:
        sw_stat, sw_pvalue = stats.shapiro(df[col].values)
        sw_values[col] = {
            "stat": sw_stat,
            "pvalue": sw_pvalue,
        }

    sw_df = pd.DataFrame.from_dict(sw_values, orient="index")
    sw_df.reset_index(inplace=True)
    sw_df.columns = ["column", "stat", "pvalue"]

    return sw_df, RawData(shapiro_results=sw_values, dataset=dataset.input_id)
