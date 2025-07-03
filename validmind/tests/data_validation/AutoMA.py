# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from validmind import RawData, tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags("time_series_data", "statsmodels", "forecasting", "statistical_test")
@tasks("regression")
def AutoMA(
    dataset: VMDataset, max_ma_order: int = 3
) -> Tuple[Dict[str, pd.DataFrame], RawData]:
    """
    Automatically selects the optimal Moving Average (MA) order for each variable in a time series dataset based on
    minimal BIC and AIC values.

    ### Purpose

    The `AutoMA` metric serves an essential role of automated decision-making for selecting the optimal Moving Average
    (MA) order for every variable in a given time series dataset. The selection is dependent on the minimalization of
    BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion); these are established statistical
    tools used for model selection. Furthermore, prior to the commencement of the model fitting process, the algorithm
    conducts a stationarity test (Augmented Dickey-Fuller test) on each series.

    ### Test Mechanism

    Starting off, the `AutoMA` algorithm checks whether the `max_ma_order` parameter has been provided. It consequently
    loops through all variables in the dataset, carrying out the Dickey-Fuller test for stationarity. For each
    stationary variable, it fits an ARIMA model for orders running from 0 to `max_ma_order`. The result is a list
    showcasing the BIC and AIC values of the ARIMA models based on different orders. The MA order, which yields the
    smallest BIC, is chosen as the 'best MA order' for every single variable. The final results include a table
    summarizing the auto MA analysis and another table listing the best MA order for each variable.

    ### Signs of High Risk

    - When a series is non-stationary (p-value>0.05 in the Dickey-Fuller test), the produced result could be inaccurate.
    - Any error that arises in the process of fitting the ARIMA models, especially with a higher MA order, can
    potentially indicate risks and might need further investigation.

    ### Strengths

    - The metric facilitates automation in the process of selecting the MA order for time series forecasting. This
    significantly saves time and reduces efforts conventionally necessary for manual hyperparameter tuning.
    - The use of both BIC and AIC enhances the likelihood of selecting the most suitable model.
    - The metric ascertains the stationarity of the series prior to model fitting, thus ensuring that the underlying
    assumptions of the MA model are fulfilled.

    ### Limitations

    - If the time series fails to be stationary, the metric may yield inaccurate results. Consequently, it necessitates
    pre-processing steps to stabilize the series before fitting the ARIMA model.
    - The metric adopts a rudimentary model selection process based on BIC and doesn't consider other potential model
    selection strategies. Depending on the specific dataset, other strategies could be more appropriate.
    - The 'max_ma_order' parameter must be manually input which doesn't always guarantee optimal performance,
    especially when configured too low.
    - The computation time increases with the rise in `max_ma_order`, hence, the metric may become computationally
    costly for larger values.
    """
    df = dataset.df

    summary_ma_analysis = pd.DataFrame()
    best_ma_order = pd.DataFrame()

    for col in df.columns:
        series = df[col].dropna()

        # Check for stationarity using the Augmented Dickey-Fuller test
        adf_test = adfuller(series)
        if adf_test[1] > 0.05:
            logger.warning(
                f"Warning: {col} is not stationary. Results may be inaccurate."
            )

        for ma_order in range(0, max_ma_order + 1):
            try:
                model = ARIMA(series, order=(0, 0, ma_order))
                model_fit = model.fit()

                # Append the result of each MA order directly into the DataFrame
                summary_ma_analysis = pd.concat(
                    [
                        summary_ma_analysis,
                        pd.DataFrame(
                            [
                                {
                                    "Variable": col,
                                    "MA Order": ma_order,
                                    "BIC": model_fit.bic,
                                    "AIC": model_fit.aic,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            except Exception as e:
                logger.error(f"Error fitting MA({ma_order}) model for {col}: {e}")

        # Find the best MA Order for this variable based on the minimum BIC
        variable_summary = summary_ma_analysis[summary_ma_analysis["Variable"] == col]
        best_bic_row = variable_summary[
            variable_summary["BIC"] == variable_summary["BIC"].min()
        ]
        best_ma_order = pd.concat([best_ma_order, best_bic_row])

    # Convert the 'MA Order' column to integer
    summary_ma_analysis["MA Order"] = summary_ma_analysis["MA Order"].astype(int)
    best_ma_order["MA Order"] = best_ma_order["MA Order"].astype(int)

    return {
        "Auto MA Analysis Results": summary_ma_analysis,
        "Best MA Order Results": best_ma_order,
    }, RawData(raw_series_data=df, dataset=dataset.input_id)
