# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from validmind import tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags(
    "time_series_data", "statsmodels", "forecasting", "statistical_test", "stationarity"
)
@tasks("regression")
def ADF(dataset: VMDataset) -> Dict[str, pd.DataFrame]:
    """
    Assesses the stationarity of a time series dataset using the Augmented Dickey-Fuller (ADF) test.

    ### Purpose

    The Augmented Dickey-Fuller (ADF) test metric is used to determine the order of integration, i.e., the stationarity
    of a given time series dataset. The stationary property of data is pivotal in many machine learning models as it
    impacts the reliability and effectiveness of predictions and forecasts.

    ### Test Mechanism

    The ADF test is executed using the `adfuller` function from the `statsmodels` library on each feature of the
    dataset. Multiple outputs are generated for each run, including the ADF test statistic and p-value, count of lags
    used, the number of observations considered in the test, critical values at various confidence levels, and the
    information criterion. These results are stored for each feature for subsequent analysis.

    ### Signs of High Risk

    - An inflated ADF statistic and high p-value (generally above 0.05) indicate a high risk to the model's performance
    due to the presence of a unit root indicating non-stationarity.
    - Non-stationarity might result in untrustworthy or insufficient forecasts.

    ### Strengths

    - The ADF test is robust to sophisticated correlations within the data, making it suitable for settings where data
    displays complex stochastic behavior.
    - It provides explicit outputs like test statistics, critical values, and information criterion, enhancing
    understanding and transparency in the model validation process.

    ### Limitations

    - The ADF test might demonstrate low statistical power, making it challenging to differentiate between a unit root
    and near-unit-root processes, potentially causing false negatives.
    - It assumes the data follows an autoregressive process, which might not always be the case.
    - The test struggles with time series data that have structural breaks.
    """
    df = dataset.df.dropna()

    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(
            "Dataset index must be a datetime or period index for time series analysis."
        )

    adf_values = {}
    for col in df.columns:
        try:
            adf_result = adfuller(df[col].values)
            adf_values[col] = {
                "ADF Statistic": adf_result[0],
                "P-Value": adf_result[1],
                "Used Lag": adf_result[2],
                "Number of Observations": adf_result[3],
                "Critical Values": adf_result[4],
                "IC Best": adf_result[5],
            }
        except Exception as e:
            logger.error(f"Error processing column '{col}': {e}")
            adf_values[col] = {
                "ADF Statistic": None,
                "P-Value": None,
                "Used Lag": None,
                "Number of Observations": None,
                "Critical Values": None,
                "IC Best": None,
                "Error": str(e),
            }

    table = pd.DataFrame.from_dict(adf_values, orient="index")
    table = table.reset_index()
    table.columns = [
        "Feature",
        "ADF Statistic",
        "P-Value",
        "Used Lag",
        "Number of Observations",
        "Critical Values",
        "IC Best",
    ]
    table = table.rename_axis("Index", axis=1)

    return {"ADF Test Results for Each Feature": table}
