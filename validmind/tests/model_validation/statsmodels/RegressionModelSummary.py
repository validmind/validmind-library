# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from sklearn.metrics import mean_squared_error, r2_score

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

from .statsutils import adj_r2_score


@tags("model_performance", "regression")
@tasks("regression")
def RegressionModelSummary(
    dataset: VMDataset, model: VMModel
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Evaluates regression model performance using metrics including R-Squared, Adjusted R-Squared, MSE, and RMSE.

    ### Purpose

    The Regression Model Summary test evaluates the performance of regression models by measuring their predictive
    ability regarding dependent variables given changes in the independent variables. It uses conventional regression
    metrics such as R-Squared, Adjusted R-Squared, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to
    assess the model's accuracy and fit.

    ### Test Mechanism

    This test uses the sklearn library to calculate the R-Squared, Adjusted R-Squared, MSE, and RMSE. It outputs a
    table with the results of these metrics along with the feature columns used by the model.

    ### Signs of High Risk

    - Low R-Squared and Adjusted R-Squared values.
    - High MSE and RMSE values.

    ### Strengths

    - Offers an extensive evaluation of regression models by combining four key measures of model accuracy and fit.
    - Provides a comprehensive view of the model's performance.
    - Both the R-Squared and Adjusted R-Squared measures are readily interpretable.

    ### Limitations

    - RMSE and MSE might be sensitive to outliers.
    - A high R-Squared or Adjusted R-Squared may not necessarily indicate a good model, especially in cases of
    overfitting.
    """
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    return [
        {
            "Independent Variables": dataset.feature_columns,
            "R-Squared": r2_score(y_true, y_pred),
            "Adjusted R-Squared": adj_r2_score(
                y_true, y_pred, len(y_true), len(dataset.feature_columns)
            ),
            "MSE": mean_squared_error(y_true, y_pred, squared=True),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        }
    ], RawData(
        y_true=y_true, y_pred=y_pred, model=model.input_id, dataset=dataset.input_id
    )
