# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from validmind import RawData, tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)


@tags("sklearn", "model_performance")
@tasks("regression")
def RegressionPerformance(
    model: VMModel, dataset: VMDataset
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Evaluates the performance of a regression model using five different metrics: MAE, MSE, RMSE, MAPE, and MBD.

    ### Purpose

    The Regression Models Performance Comparison metric is used to measure the performance of regression models. It
    calculates multiple evaluation metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Mean Bias Deviation (MBD), thereby
    enabling a comprehensive view of model performance.

    ### Test Mechanism

    The test uses the sklearn library to calculate the MAE, MSE, RMSE, MAPE, and MBD. These calculations encapsulate both
    the direction and the magnitude of error in predictions, thereby providing a multi-faceted view of model accuracy.

    ### Signs of High Risk

    - High values of MAE, MSE, RMSE, and MAPE, which indicate a high error rate and imply a larger departure of the
    model's predictions from the true values.
    - A large value of MBD, which shows a consistent bias in the model’s predictions.

    ### Strengths

    - The metric evaluates models on five different metrics offering a comprehensive analysis of model performance.
    - It is designed to handle regression tasks and can be seamlessly integrated with libraries like sklearn.

    ### Limitations

    - The metric only evaluates regression models and does not evaluate classification models.
    - The test assumes that the models have been trained and tested appropriately prior to evaluation. It does not
    handle pre-processing, feature selection, or other stages in the model lifecycle.
    """
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # MAE calculation
    metrics = {
        "Mean Absolute Error (MAE)": mean_absolute_error(y_true, y_pred),
    }

    # MSE and RMSE calculations
    mse = mean_squared_error(y_true, y_pred)
    metrics["Mean Squared Error (MSE)"] = mse
    metrics["Root Mean Squared Error (RMSE)"] = np.sqrt(mse)

    # MAPE calculation
    if np.any(y_true == 0):
        logger.warning(
            "y_true contains zero values. Skipping MAPE calculation to avoid division by zero."
        )
        metrics["Mean Absolute Percentage Error (MAPE)"] = None
    else:
        metrics["Mean Absolute Percentage Error (MAPE)"] = (
            np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        )

    # MBD calculation
    metrics["Mean Bias Deviation (MBD)"] = np.mean(y_pred - y_true)

    return (
        [
            {
                "Metric": metric,
                "Value": value,
            }
            for metric, value in metrics.items()
        ],
        RawData(
            y_true=y_true, y_pred=y_pred, model=model.input_id, dataset=dataset.input_id
        ),
    )
