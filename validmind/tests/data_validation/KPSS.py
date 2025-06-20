# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict

import pandas as pd
from statsmodels.tsa.stattools import kpss

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags("time_series_data", "stationarity", "unit_root_test", "statsmodels")
@tasks("data_validation")
def KPSS(dataset: VMDataset) -> Dict[str, Any]:
    """
    Assesses the stationarity of time-series data in a machine learning model using the KPSS unit root test.

    ### Purpose

    The KPSS (Kwiatkowski-Phillips-Schmidt-Shin) unit root test is utilized to ensure the stationarity of data within a
    machine learning model. It specifically works on time-series data to establish the order of integration, which is
    essential for accurate forecasting. A fundamental requirement for any time series model is that the series should
    be stationary.

    ### Test Mechanism

    This test calculates the KPSS score for each feature in the dataset. The KPSS score includes a statistic, a
    p-value, a used lag, and critical values. The core principle behind the KPSS test is to evaluate the hypothesis
    that an observable time series is stationary around a deterministic trend. If the computed statistic exceeds the
    critical value, the null hypothesis (that the series is stationary) is rejected, indicating that the series is
    non-stationary.

    ### Signs of High Risk

    - High KPSS score, particularly if the calculated statistic is higher than the critical value.
    - Rejection of the null hypothesis, indicating that the series is recognized as non-stationary, can severely affect
    the model's forecasting capability.

    ### Strengths

    - Directly measures the stationarity of a series, fulfilling a key prerequisite for many time-series models.
    - The underlying logic of the test is intuitive and simple, making it easy to understand and accessible for both
    developers and risk management teams.

    ### Limitations

    - Assumes the absence of a unit root in the series and doesn't differentiate between series that are stationary and
    those border-lining stationarity.
    - The test may have restricted power against certain alternatives.
    - The reliability of the test is contingent on the number of lags selected, which introduces potential bias in the
    measurement.
    """
    df = dataset.df.dropna()

    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise SkipTestError(
            "Dataset index must be a datetime or period index for time series analysis."
        )

    df = df.apply(pd.to_numeric, errors="coerce")

    kpss_table = []

    for col in dataset.columns:
        kpss_stat, pvalue, usedlag, critical_values = kpss(df[col].values)
        kpss_table.append(
            {
                "Variable": col,
                "stat": kpss_stat,
                "pvalue": pvalue,
                "usedlag": usedlag,
                "critical_values": critical_values,
            }
        )

    if not kpss_table:
        raise SkipTestError(f"No KPSS results found for dataset: {dataset.input_id}")

    return {
        "KPSS Test Results": kpss_table,
    }
