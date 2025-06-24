# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

from validmind import tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags("time_series_data", "statsmodels", "forecasting", "statistical_test")
@tasks("regression")
def AutoAR(dataset: VMDataset, max_ar_order: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Automatically identifies the optimal Autoregressive (AR) order for a time series using BIC and AIC criteria.

    ### Purpose

    The AutoAR test is intended to automatically identify the Autoregressive (AR) order of a time series by utilizing
    the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC). AR order is crucial in forecasting
    tasks as it dictates the quantity of prior terms in the sequence to use for predicting the current term. The
    objective is to select the most fitting AR model that encapsulates the trend and seasonality in the time series
    data.

    ### Test Mechanism

    The test mechanism operates by iterating through a possible range of AR orders up to a defined maximum. An AR model
    is fitted for each order, and the corresponding BIC and AIC are computed. BIC and AIC statistical measures are
    designed to penalize models for complexity, preferring simpler models that fit the data proficiently. To verify the
    stationarity of the time series, the Augmented Dickey-Fuller test is executed. The AR order, BIC, and AIC findings
    are compiled into a dataframe for effortless comparison. Then, the AR order with the smallest BIC is established as
    the desirable order for each variable.

    ### Signs of High Risk

    - An augmented Dickey Fuller test p-value > 0.05, indicating the time series isn't stationary, may lead to
    inaccurate results.
    - Problems with the model fitting procedure, such as computational or convergence issues.
    - Continuous selection of the maximum specified AR order may suggest an insufficient set limit.

    ### Strengths

    - The test independently pinpoints the optimal AR order, thereby reducing potential human bias.
    - It strikes a balance between model simplicity and goodness-of-fit to avoid overfitting.
    - Has the capability to account for stationarity in a time series, an essential aspect for dependable AR modeling.
    - The results are aggregated into a comprehensive table, enabling an easy interpretation.

    ### Limitations

    - The tests need a stationary time series input.
    - They presume a linear relationship between the series and its lags.
    - The search for the best model is constrained by the maximum AR order supplied in the parameters. Therefore, a low
    max_ar_order could result in subpar outcomes.
    - AIC and BIC may not always agree on the selection of the best model. This potentially requires the user to juggle
    interpretational choices.
    """
    df = dataset.df

    summary_ar_analysis = pd.DataFrame()
    best_ar_order = pd.DataFrame()

    for col in df.columns:
        series = df[col].dropna()

        # Check for stationarity using the Augmented Dickey-Fuller test
        adf_test = adfuller(series)
        if adf_test[1] > 0.05:
            logger.warning(
                f"Warning: {col} is not stationary. Results may be inaccurate."
            )

        for ar_order in range(0, max_ar_order + 1):
            try:
                model = AutoReg(series, lags=ar_order, old_names=False)
                model_fit = model.fit()

                # Append the result of each AR order directly into the DataFrame
                summary_ar_analysis = pd.concat(
                    [
                        summary_ar_analysis,
                        pd.DataFrame(
                            [
                                {
                                    "Variable": col,
                                    "AR Order": ar_order,
                                    "BIC": model_fit.bic,
                                    "AIC": model_fit.aic,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            except Exception as e:
                logger.error(f"Error fitting AR({ar_order}) model for {col}: {e}")

        # Find the best AR Order for this variable based on the minimum BIC
        variable_summary = summary_ar_analysis[summary_ar_analysis["Variable"] == col]
        best_bic_row = variable_summary[
            variable_summary["BIC"] == variable_summary["BIC"].min()
        ]
        best_ar_order = pd.concat([best_ar_order, best_bic_row])

    # Convert the 'AR Order' column to integer
    summary_ar_analysis["AR Order"] = summary_ar_analysis["AR Order"].astype(int)
    best_ar_order["AR Order"] = best_ar_order["AR Order"].astype(int)

    return {
        "Auto AR Analysis Results": summary_ar_analysis,
        "Best AR Order Results": best_ar_order,
    }
