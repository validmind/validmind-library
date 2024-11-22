# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Union

from sklearn.model_selection import GridSearchCV

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance")
@tasks("classification", "clustering")
def HyperParametersTuning(
    model: VMModel,
    dataset: VMDataset,
    param_grid: Union[dict, None] = None,
    scoring: Union[str, None] = None,
):
    """
    Exerts exhaustive grid search to identify optimal hyperparameters for the model, improving performance.

    ### Purpose:

    The "HyperParametersTuning" metric aims to find the optimal set of hyperparameters for a given model. The test is
    designed to enhance the performance of the model by determining the best configuration of hyperparameters. The
    parameters that are being optimized are defined by the parameter grid provided to the metric.

    ### Test Mechanism:

    The HyperParametersTuning test employs a grid search mechanism using the GridSearchCV function from the
    scikit-learn library. The grid search algorithm systematically works through multiple combinations of parameter
    values, cross-validating to determine which combination gives the best model performance. The chosen model and the
    parameter grid passed for tuning are necessary inputs. Once the grid search is complete, the test caches and
    returns details of the best model and its associated parameters.

    ### Signs of High Risk:

    - The test raises a SkipTestError if the param_grid is not supplied, indicating a lack of specific parameters to
    optimize, which can be risky for certain model types reliant on parameter tuning.
    - Poorly chosen scoring metrics that do not align well with the specific model or problem at hand could reflect
    potential risks or failures in achieving optimal performance.

    ### Strengths:

    - Provides a comprehensive exploration mechanism to identify the best set of hyperparameters for the supplied
    model, thereby enhancing its performance.
    - Implements GridSearchCV, simplifying and automating the time-consuming task of hyperparameter tuning.

    ### Limitations:

    - The grid search algorithm can be computationally expensive, especially with large datasets or complex models, and
    can be time-consuming as it tests all possible combinations within the specified parameter grid.
    - The effectiveness of the tuning is heavily dependent on the quality of data and only accepts datasets with
    numerical or ordered categories.
    - Assumes that the same set of hyperparameters is optimal for all problem sets, which may not be true in every
    scenario.
    - There's a potential risk of overfitting the model if the training set is not representative of the data that the
    model will be applied to.
    """
    if not param_grid:
        raise SkipTestError("'param_grid' dictionary must be provided to run this test")

    estimators = GridSearchCV(model.model, param_grid=param_grid, scoring=scoring)
    estimators.fit(dataset.x, dataset.y)

    return [
        {
            "Best Model": estimators.best_estimator_,
            "Best Parameters": estimators.best_params_,
        }
    ]
