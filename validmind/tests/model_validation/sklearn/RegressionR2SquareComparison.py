# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List

import pandas as pd
from sklearn import metrics

from validmind import tags, tasks
from validmind.tests.model_validation.statsmodels.statsutils import adj_r2_score
from validmind.vm_models import VMDataset, VMModel


@tags("model_performance", "sklearn")
@tasks("regression", "time_series_forecasting")
def RegressionR2SquareComparison(
    datasets: List[VMDataset], models: List[VMModel]
) -> pd.DataFrame:
    """
    Compares R-Squared and Adjusted R-Squared values for different regression models across multiple datasets to assess
    model performance and relevance of features.

    ### Purpose

    The Regression R2 Square Comparison test aims to compare the R-Squared and Adjusted R-Squared values for different
    regression models across various datasets. It helps in assessing how well each model explains the variability in
    the dataset, and whether the models include irrelevant features.

    ### Test Mechanism

    This test operates by:

    - Iterating through each dataset-model pair.
    - Calculating the R-Squared values to measure how much of the variability in the dataset is explained by the model.
    - Calculating the Adjusted R-Squared values, which adjust the R-Squared based on the number of predictors in the
    model, making it more reliable when comparing models with different numbers of features.
    - Generating a summary table containing these values for each combination of dataset and model.

    ### Signs of High Risk

    - If the R-Squared values are significantly low, it indicates the model isn't explaining much of the variability in
    the dataset.
    - A significant difference between R-Squared and Adjusted R-Squared values might indicate that the model includes
    irrelevant features.

    ### Strengths

    - Provides a quantitative measure of model performance in terms of variance explained.
    - Adjusted R-Squared accounts for the number of predictors, making it a more reliable measure when comparing models
    with different numbers of features.
    - Useful for time-series forecasting and regression tasks.

    ### Limitations

    - Assumes the dataset is provided as a DataFrameDataset object with `y`, `y_pred`, and `feature_columns` attributes.
    - Relies on `adj_r2_score` from the `statsmodels.statsutils` module, which needs to be correctly implemented and
    imported.
    - Requires that `dataset.y_pred(model)` returns the predicted values for the model.
    """
    results_list = []

    for dataset, model in zip(datasets, models):
        dataset_name = dataset.input_id
        model_name = model.input_id

        y_true = dataset.y
        y_pred = dataset.y_pred(model)  # Assuming dataset has X for features
        y_true = y_true.astype(y_pred.dtype)

        r2s = metrics.r2_score(y_true, y_pred)
        X_columns = dataset.feature_columns
        adj_r2 = adj_r2_score(y_true, y_pred, len(y_true), len(X_columns))

        # Append results to the list
        results_list.append(
            {
                "Model": model_name,
                "Dataset": dataset_name,
                "R-Squared": r2s,
                "Adjusted R-Squared": adj_r2,
            }
        )

    return pd.DataFrame(results_list)
