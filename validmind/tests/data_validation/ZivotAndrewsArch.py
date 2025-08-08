# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, Tuple

import pandas as pd
from numpy.linalg import LinAlgError

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError, SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

try:
    from arch.unitroot import ZivotAndrews
except ImportError as e:
    if "arch" in str(e):
        raise MissingDependencyError(
            "Missing required package `arch` for ZivotAndrewsArch. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["arch"],
            extra="stats",
        ) from e
    raise e

logger = get_logger(__name__)


@tags("time_series_data", "stationarity", "unit_root_test")
@tasks("regression")
def ZivotAndrewsArch(dataset: VMDataset) -> Tuple[Dict[str, Any], RawData]:
    """
    Evaluates the order of integration and stationarity of time series data using the Zivot-Andrews unit root test.

    ### Purpose

    The Zivot-Andrews Arch metric is used to evaluate the order of integration for time series data in a machine
    learning model. It's designed to test for stationarity, a crucial aspect of time series analysis, where data points
    are independent of time. Stationarity means that the statistical properties such as mean, variance, and
    autocorrelation are constant over time.

    ### Test Mechanism

    The Zivot-Andrews unit root test is performed on each feature in the dataset using the `ZivotAndrews` function from
    the `arch.unitroot` module. This function returns several metrics for each feature, including the statistical
    value, p-value (probability value), the number of lags used, and the number of observations. The p-value is used to
    decide on the null hypothesis (the time series has a unit root and is non-stationary) based on a chosen level of
    significance.

    ### Signs of High Risk

    - A high p-value suggests high risk, indicating insufficient evidence to reject the null hypothesis, implying that
    the time series has a unit root and is non-stationary.
    - Non-stationary time series data can lead to misleading statistics and unreliable machine learning models.

    ### Strengths

    - Dynamically tests for stationarity against structural breaks in time series data, offering robust evaluation of
    stationarity in features.
    - Especially beneficial with financial, economic, or other time-series data where data observations lack a
    consistent pattern and structural breaks may occur.

    ### Limitations

    - Assumes data is derived from a single-equation, autoregressive model, making it less appropriate for multivariate
    time series data or data not aligning with this model.
    - May not account for unexpected shocks or changes in the series trend, both of which can significantly impact data
    stationarity.
    """
    df = dataset.df
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise SkipTestError(
            "Dataset index must be a datetime or period index for time series analysis."
        )

    df = df.dropna()
    df = df.apply(pd.to_numeric, errors="coerce")

    za_values = []

    for col in df.columns:
        try:
            za = ZivotAndrews(df[col].values)
        except (LinAlgError, ValueError) as e:
            logger.error(f"Error while processing column '{col}': {e}")
            continue

        za_values.append(
            {
                "Variable": col,
                "stat": za.stat,
                "pvalue": za.pvalue,
                "usedlag": za.lags,
                "nobs": za.nobs,
            }
        )

    return (
        {"Zivot-Andrews Test Results": za_values},
        RawData(zivot_andrews=za_values, dataset=dataset.input_id),
    )
