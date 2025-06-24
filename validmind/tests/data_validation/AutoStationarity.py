# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from validmind import tags, tasks
from validmind.vm_models import VMDataset


@tags("time_series_data", "statsmodels", "forecasting", "statistical_test")
@tasks("regression")
def AutoStationarity(
    dataset: VMDataset, max_order: int = 5, threshold: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Automates Augmented Dickey-Fuller test to assess stationarity across multiple time series in a DataFrame.

    ### Purpose

    The AutoStationarity metric is intended to automatically detect and evaluate the stationary nature of each time
    series in a DataFrame. It incorporates the Augmented Dickey-Fuller (ADF) test, a statistical approach used to
    assess stationarity. Stationarity is a fundamental property suggesting that statistic features like mean and
    variance remain unchanged over time. This is necessary for many time-series models.

    ### Test Mechanism

    The mechanism for the AutoStationarity test involves applying the Augmented Dicky-Fuller test to each time series
    within the given dataframe to assess if they are stationary. Every series in the dataframe is looped, using the ADF
    test up to a defined maximum order (configurable and by default set to 5). The p-value resulting from the ADF test
    is compared against a predetermined threshold (also configurable and by default set to 0.05). The time series is
    deemed stationary at its current differencing order if the p-value is less than the threshold.

    ### Signs of High Risk

    - A significant number of series not achieving stationarity even at the maximum order of differencing can indicate
    high risk or potential failure in the model.
    - This could suggest the series may not be appropriately modeled by a stationary process, hence other modeling
    approaches might be required.

    ### Strengths

    - The key strength in this metric lies in the automation of the ADF test, enabling mass stationarity analysis
    across various time series and boosting the efficiency and credibility of the analysis.
    - The utilization of the ADF test, a widely accepted method for testing stationarity, lends authenticity to the
    results derived.
    - The introduction of the max order and threshold parameters give users the autonomy to determine their preferred
    levels of stringency in the tests.

    ### Limitations

    - The Augmented Dickey-Fuller test and the stationarity test are not without their limitations. These tests are
    premised on the assumption that the series can be modeled by an autoregressive process, which may not always hold
    true.
    - The stationarity check is highly sensitive to the choice of threshold for the significance level; an extremely
    high or low threshold could lead to incorrect results regarding the stationarity properties.
    - There's also a risk of over-differencing if the maximum order is set too high, which could induce unnecessary
    cycles.
    """
    df = dataset.df.dropna()

    summary_stationarity = pd.DataFrame()
    best_integration_order = pd.DataFrame()

    # Loop over each column in the input DataFrame and perform stationarity tests
    for col in df.columns:
        is_stationary = False
        order = 0

        while not is_stationary and order <= max_order:
            series = df[col]

            if order == 0:
                adf_result = adfuller(series)
            else:
                adf_result = adfuller(np.diff(series, n=order))

            adf_pvalue = adf_result[1]
            adf_pass_fail = adf_pvalue < threshold
            adf_decision = "Stationary" if adf_pass_fail else "Non-stationary"

            # Append the result of each test directly into the DataFrame
            summary_stationarity = pd.concat(
                [
                    summary_stationarity,
                    pd.DataFrame(
                        [
                            {
                                "Variable": col,
                                "Integration Order": order,
                                "Test": "ADF",
                                "p-value": adf_pvalue,
                                "Threshold": threshold,
                                "Pass/Fail": "Pass" if adf_pass_fail else "Fail",
                                "Decision": adf_decision,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            if adf_pass_fail:
                is_stationary = True
                best_integration_order = pd.concat(
                    [
                        best_integration_order,
                        pd.DataFrame(
                            [
                                {
                                    "Variable": col,
                                    "Best Integration Order": order,
                                    "Test": "ADF",
                                    "p-value": adf_pvalue,
                                    "Threshold": threshold,
                                    "Decision": adf_decision,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            order += 1

    # Convert the 'Integration Order' and 'Best Integration Order' column to integer
    summary_stationarity["Integration Order"] = summary_stationarity[
        "Integration Order"
    ].astype(int)
    best_integration_order["Best Integration Order"] = best_integration_order[
        "Best Integration Order"
    ].astype(int)

    return {
        "Stationarity Analysis Results": summary_stationarity,
        "Best Integration Order Results": best_integration_order,
    }
