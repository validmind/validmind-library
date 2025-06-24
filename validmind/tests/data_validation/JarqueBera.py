# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
from statsmodels.stats.stattools import jarque_bera

from validmind import RawData, tags, tasks


@tasks("classification", "regression")
@tags("tabular_data", "data_distribution", "statistical_test", "statsmodels")
def JarqueBera(dataset) -> Tuple[pd.DataFrame, RawData]:
    """
    Assesses normality of dataset features in an ML model using the Jarque-Bera test.

    ### Purpose

    The purpose of the Jarque-Bera test as implemented in this metric is to determine if the features in the dataset of
    a given Machine Learning model follow a normal distribution. This is crucial for understanding the distribution and
    behavior of the model's features, as numerous statistical methods assume normal distribution of the data.

    ### Test Mechanism

    The test mechanism involves computing the Jarque-Bera statistic, p-value, skew, and kurtosis for each feature in
    the dataset. It utilizes the 'jarque_bera' function from the 'statsmodels' library in Python, storing the results
    in a dictionary. The test evaluates the skewness and kurtosis to ascertain whether the dataset follows a normal
    distribution. A significant p-value (typically less than 0.05) implies that the data does not possess normal
    distribution.

    ### Signs of High Risk

    - A high Jarque-Bera statistic and a low p-value (usually less than 0.05) indicate high-risk conditions.
    - Such results suggest the data significantly deviates from a normal distribution. If a machine learning model
    expects feature data to be normally distributed, these findings imply that it may not function as intended.

    ### Strengths

    - Provides insights into the shape of the data distribution, helping determine whether a given set of data follows
    a normal distribution.
    - Particularly useful for risk assessment for models that assume a normal distribution of data.
    - By measuring skewness and kurtosis, it provides additional insights into the nature and magnitude of a
    distribution's deviation.

    ### Limitations

    - Only checks for normality in the data distribution. It cannot provide insights into other types of distributions.
    - Datasets that aren't normally distributed but follow some other distribution might lead to inaccurate risk
    assessments.
    - Highly sensitive to large sample sizes, often rejecting the null hypothesis (that data is normally distributed)
    even for minor deviations in larger datasets.
    """

    df = dataset.df[dataset.feature_columns_numeric]

    jb_values = {}
    for col in df.columns:
        jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(df[col].values)
        jb_values[col] = {
            "stat": jb_stat,
            "pvalue": jb_pvalue,
            "skew": jb_skew,
            "kurtosis": jb_kurtosis,
        }

    jb_df = pd.DataFrame.from_dict(jb_values, orient="index")
    jb_df.reset_index(inplace=True)
    jb_df.columns = ["column", "stat", "pvalue", "skew", "kurtosis"]

    return jb_df, RawData(jb_values=jb_values, dataset=dataset.input_id)
