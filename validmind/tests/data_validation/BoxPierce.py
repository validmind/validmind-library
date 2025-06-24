# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from validmind import RawData, tags, tasks


@tasks("regression")
@tags("time_series_data", "forecasting", "statistical_test", "statsmodels")
def BoxPierce(dataset) -> Tuple[pd.DataFrame, RawData]:
    """
    Detects autocorrelation in time-series data through the Box-Pierce test to validate model performance.

    ### Purpose

    The Box-Pierce test is utilized to detect the presence of autocorrelation in a time-series dataset.
    Autocorrelation, or serial correlation, refers to the degree of similarity between observations based on the
    temporal spacing between them. This test is essential for affirming the quality of a time-series model by ensuring
    that the error terms in the model are random and do not adhere to a specific pattern.

    ### Test Mechanism

    The implementation of the Box-Pierce test involves calculating a test statistic along with a corresponding p-value
    derived from the dataset features. These quantities are used to test the null hypothesis that posits the data to be
    independently distributed. This is achieved by iterating over every feature column in the time-series data and
    applying the `acorr_ljungbox` function of the statsmodels library. The function yields the Box-Pierce test
    statistic as well as the respective p-value, all of which are cached as test results.

    ### Signs of High Risk

    - A low p-value, typically under 0.05 as per statistical convention, throws the null hypothesis of independence
    into question. This implies that the dataset potentially houses autocorrelations, thus indicating a high-risk
    scenario concerning model performance.
    - Large Box-Pierce test statistic values may indicate the presence of autocorrelation.

    ### Strengths

    - Detects patterns in data that are supposed to be random, thereby ensuring no underlying autocorrelation.
    - Can be computed efficiently given its low computational complexity.
    - Can be widely applied to most regression problems, making it very versatile.

    ### Limitations

    - Assumes homoscedasticity (constant variance) and normality of residuals, which may not always be the case in
    real-world datasets.
    - May exhibit reduced power for detecting complex autocorrelation schemes such as higher-order or negative
    correlations.
    - It only provides a general indication of the existence of autocorrelation, without providing specific insights
    into the nature or patterns of the detected autocorrelation.
    - In the presence of trends or seasonal patterns, the Box-Pierce test may yield misleading results.
    - Applicability is limited to time-series data, which limits its overall utility.
    """

    df = dataset.df

    box_pierce_values = {}
    for col in df.columns:
        bp_results = acorr_ljungbox(df[col].values, boxpierce=True, return_df=True)
        box_pierce_values[col] = {
            "stat": bp_results.iloc[0]["lb_stat"],
            "pvalue": bp_results.iloc[0]["lb_pvalue"],
        }

    box_pierce_df = pd.DataFrame.from_dict(box_pierce_values, orient="index")
    box_pierce_df.reset_index(inplace=True)
    box_pierce_df.columns = ["column", "stat", "pvalue"]

    return box_pierce_df, RawData(
        box_pierce_values=box_pierce_values, dataset=dataset.input_id
    )
