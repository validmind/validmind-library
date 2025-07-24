# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from validmind import tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)


@tags("time_series_data", "forecasting", "model_selection", "statsmodels")
@tasks("regression")
def AutoARIMA(model: VMModel, dataset: VMDataset) -> List[Dict[str, float]]:
    """
    Evaluates ARIMA models for time-series forecasting, ranking them using Bayesian and Akaike Information Criteria.

    ### Purpose

    The AutoARIMA validation test is designed to evaluate and rank AutoRegressive Integrated Moving Average (ARIMA)
    models. These models are primarily used for forecasting time-series data. The validation test automatically fits
    multiple ARIMA models, with varying parameters, to every variable within the given dataset. The models are then
    ranked based on their Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) values, which
    provide a basis for the efficient model selection process.

    ### Test Mechanism

    This metric proceeds by generating an array of feasible combinations of ARIMA model parameters which are within a
    prescribed limit. These limits include `max_p`, `max_d`, `max_q`; they represent the autoregressive, differencing,
    and moving average components respectively. Upon applying these sets of parameters, the validation test fits each
    ARIMA model to the time-series data provided. For each model, it subsequently proceeds to calculate and record both
    the BIC and AIC values, which serve as performance indicators for the model fit. Prior to this parameter fitting
    process, the Augmented Dickey-Fuller test for data stationarity is conducted on the data series. If a series is
    found to be non-stationary, a warning message is sent out, given that ARIMA models necessitate input series to be
    stationary.

    ### Signs of High Risk

    - If the p-value of the Augmented Dickey-Fuller test for a variable exceeds 0.05, a warning is logged. This warning
    indicates that the series might not be stationary, leading to potentially inaccurate results.
    - Consistent failure in fitting ARIMA models (as made evident through logged errors) might disclose issues with
    either the data or model stability.

    ### Strengths

    - The AutoARIMA validation test simplifies the often complex task of selecting the most suitable ARIMA model based
    on BIC and AIC criteria.
    - The mechanism incorporates a check for non-stationarity within the data, which is a critical prerequisite for
    ARIMA models.
    - The exhaustive search through all possible combinations of model parameters enhances the likelihood of
    identifying the best-fit model.

    ### Limitations

    - This validation test can be computationally costly as it involves creating and fitting multiple ARIMA models for
    every variable.
    - Although the test checks for non-stationarity and logs warnings where present, it does not apply any
    transformations to the data to establish stationarity.
    - The selection of models leans solely on BIC and AIC criteria, which may not yield the best predictive model in
    all scenarios.
    - The test is only applicable to regression tasks involving time-series data, and may not work effectively for
    other types of machine learning tasks.
    """
    max_p = 3
    max_d = 2
    max_q = 3

    df = dataset.x_df()

    table = []

    for col in df.columns:
        series = df[col].dropna()

        # Check for stationarity using the Augmented Dickey-Fuller test
        adf_test = adfuller(series)
        if adf_test[1] > 0.05:
            logger.warning(f"{col} is not stationary. Results may be inaccurate.")

        arima_orders = []
        bic_values = []
        aic_values = []

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        model_fit = model.fit()

                        arima_orders.append((p, d, q))
                        bic_values.append(model_fit.bic)
                        aic_values.append(model_fit.aic)
                    except Exception as e:
                        logger.error(
                            f"Error fitting ARIMA({p}, {d}, {q}) model for {col}: {e}"
                        )

        table.append(
            {
                "Variable": col,
                "ARIMA Orders": arima_orders,
                "BIC": bic_values,
                "AIC": aic_values,
            }
        )

    return table
