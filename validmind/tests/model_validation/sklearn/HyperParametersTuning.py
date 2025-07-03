# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple, Union

from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


def custom_recall(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return recall_score(y_true, y_pred)


def _get_metrics(scoring):
    """Convert scoring parameter to list of metrics."""
    if scoring is None:
        return ["accuracy"]
    return (
        scoring
        if isinstance(scoring, list)
        else list(scoring.keys())
        if isinstance(scoring, dict)
        else [scoring]
    )


def _get_thresholds(thresholds):
    """Convert thresholds parameter to list."""
    if thresholds is None:
        return [0.5]
    return [thresholds] if isinstance(thresholds, (int, float)) else thresholds


def _create_scoring_dict(scoring, metrics, threshold):
    """Create scoring dictionary for GridSearchCV."""
    if scoring is None:
        return None

    scoring_dict = {}
    for metric in metrics:
        if metric == "recall":
            scoring_dict[metric] = make_scorer(
                custom_recall, needs_proba=True, threshold=threshold
            )
        elif metric == "roc_auc":
            scoring_dict[metric] = "roc_auc"
        else:
            scoring_dict[metric] = metric
    return scoring_dict


@tags("sklearn", "model_performance")
@tasks("clustering", "classification")
def HyperParametersTuning(
    model: VMModel,
    dataset: VMDataset,
    param_grid: dict,
    scoring: Union[str, List, Dict] = None,
    thresholds: Union[float, List[float]] = None,
    fit_params: dict = None,
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Performs exhaustive grid search over specified parameter ranges to find optimal model configurations
    across different metrics and decision thresholds.

    ### Purpose

    The Hyperparameter Tuning test systematically explores the model's parameter space to identify optimal
    configurations. It supports multiple optimization metrics and decision thresholds, providing a comprehensive
    view of how different parameter combinations affect various aspects of model performance.

    ### Test Mechanism

    The test uses scikit-learn's GridSearchCV to perform cross-validation for each parameter combination.
    For each specified threshold and optimization metric, it creates a scoring dictionary with
    threshold-adjusted metrics, performs grid search with cross-validation, records best parameters and
    corresponding scores, and combines results into a comparative table. This process is repeated for each
    optimization metric to provide a comprehensive view of model performance under different configurations.

    ### Signs of High Risk

    - Large performance variations across different parameter combinations
    - Significant discrepancies between different optimization metrics
    - Best parameters at the edges of the parameter grid
    - Unstable performance across different thresholds
    - Overly complex model configurations (risk of overfitting)
    - Very different optimal parameters for different metrics
    - Cross-validation scores showing high variance
    - Extreme parameter values in best configurations

    ### Strengths

    - Comprehensive exploration of parameter space
    - Supports multiple optimization metrics
    - Allows threshold optimization
    - Provides comparative view across different configurations
    - Uses cross-validation for robust evaluation
    - Helps understand trade-offs between different metrics
    - Enables systematic parameter selection
    - Supports both classification and clustering tasks

    ### Limitations

    - Computationally expensive for large parameter grids
    - May not find global optimum (limited to grid points)
    - Cannot handle dependencies between parameters
    - Memory intensive for large datasets
    - Limited to scikit-learn compatible models
    - Cross-validation splits may not preserve time series structure
    - Grid search may miss optimal values between grid points
    - Resource intensive for high-dimensional parameter spaces
    """
    fit_params = fit_params or {}

    # Simple case: no scoring and no thresholds
    if scoring is None and thresholds is None:
        estimators = GridSearchCV(model.model, param_grid=param_grid, scoring=None)
        estimators.fit(dataset.x_df(), dataset.y, **fit_params)
        return [
            {
                "Best Model": estimators.best_estimator_,
                "Best Parameters": estimators.best_params_,
            }
        ]

    # Complex case: with scoring or thresholds
    results = []
    metrics = _get_metrics(scoring)
    thresholds = _get_thresholds(thresholds)

    for threshold in thresholds:
        scoring_dict = _create_scoring_dict(scoring, metrics, threshold)

        for optimize_for in metrics:
            estimators = GridSearchCV(
                model.model,
                param_grid=param_grid,
                scoring=scoring_dict,
                refit=optimize_for if scoring is not None else True,
            )

            estimators.fit(dataset.x_df(), dataset.y, **fit_params)

            best_index = estimators.best_index_
            row_result = {
                "Optimized for": optimize_for,
                "Threshold": threshold,
                "Best Parameters": estimators.best_params_,
            }

            score_key = (
                "mean_test_score" if scoring is None else f"mean_test_{optimize_for}"
            )
            row_result[optimize_for] = estimators.cv_results_[score_key][best_index]

            results.append(row_result)

    return results, RawData(
        model=model.input_id, dataset=dataset.input_id, param_grid=param_grid
    )
