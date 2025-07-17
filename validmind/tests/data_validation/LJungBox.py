# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from validmind import RawData, tags, tasks


@tasks("regression")
@tags("time_series_data", "forecasting", "statistical_test", "statsmodels")
def LJungBox(dataset) -> Tuple[pd.DataFrame, RawData]:
    """
    Assesses autocorrelations in dataset features by performing a Ljung-Box test on each feature.

    ### Purpose

    The Ljung-Box test is a type of statistical test utilized to ascertain whether there are autocorrelations within a
    given dataset that differ significantly from zero. In the context of a machine learning model, this test is
    primarily used to evaluate data utilized in regression tasks, especially those involving time series and
    forecasting.

    ### Test Mechanism

    The test operates by iterating over each feature within the dataset and applying the `acorr_ljungbox`
    function from the `statsmodels.stats.diagnostic` library. This function calculates the Ljung-Box statistic and
    p-value for each feature. These results are then stored in a pandas DataFrame where the columns are the feature names,
    statistic, and p-value respectively. Generally, a lower p-value indicates a higher likelihood of significant
    autocorrelations within the feature.

    ### Signs of High Risk

    - High Ljung-Box statistic values or low p-values.
    - Presence of significant autocorrelations in the respective features.
    - Potential for negative impact on model performance or bias if autocorrelations are not properly handled.

    ### Strengths

    - Powerful tool for detecting autocorrelations within datasets, especially in time series data.
    - Provides quantitative measures (statistic and p-value) for precise evaluation.
    - Helps avoid issues related to autoregressive residuals and other challenges in regression models.

    ### Limitations

    - Cannot detect all types of non-linearity or complex interrelationships among variables.
    - Testing individual features may not fully encapsulate the dynamics of the data if features interact with each other.
    - Designed more for traditional statistical models and may not be fully compatible with certain types of complex
      machine learning models.
    """

    df = dataset.df

    ljung_box_values = {}
    for col in df.columns:
        lb_results = acorr_ljungbox(df[col].values, return_df=True)
        ljung_box_values[col] = {
            "stat": lb_results.iloc[0]["lb_stat"],
            "pvalue": lb_results.iloc[0]["lb_pvalue"],
        }

    ljung_box_df = pd.DataFrame.from_dict(ljung_box_values, orient="index")
    ljung_box_df.reset_index(inplace=True)
    ljung_box_df.columns = ["column", "stat", "pvalue"]

    return ljung_box_df, RawData(ljung_box_df=ljung_box_df, dataset=dataset.input_id)
