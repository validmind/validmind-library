# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Union, Dict, List
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance")
@tasks("classification", "clustering")
def custom_recall(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return recall_score(y_true, y_pred)


@tags("sklearn", "model_performance")
@tasks("clustering", "classification")
def HyperParametersTuning(
    model: VMModel,
    dataset: VMDataset,
    param_grid: dict,
    scoring: Union[str, List, Dict] = None,
    thresholds: Union[float, List[float]] = None,
    fit_params: dict = None,
):
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
    results = []

    # Handle default scoring
    if scoring is None:
        scoring = "accuracy"  # Default to accuracy as the scoring metric

    metrics = (
        scoring
        if isinstance(scoring, list)
        else list(scoring.keys()) if isinstance(scoring, dict) else [scoring]
    )

    # Handle default threshold
    if thresholds is None:
        thresholds = [0.5]  # Default to standard 0.5 threshold
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    # For each threshold
    for threshold in thresholds:
        # Create scoring dict with current threshold
        scoring_dict = {}
        if scoring is not None:  # Only create scoring_dict if scoring was provided
            for metric in metrics:
                if metric == "recall":
                    scoring_dict[metric] = make_scorer(
                        custom_recall, needs_proba=True, threshold=threshold
                    )
                elif metric == "roc_auc":
                    scoring_dict[metric] = "roc_auc"  # threshold independent
                else:
                    scoring_dict[metric] = metric

        # Run GridSearchCV for each optimization metric
        for optimize_for in metrics:
            estimators = GridSearchCV(
                model.model,
                param_grid=param_grid,
                scoring=(
                    scoring_dict if scoring_dict else None
                ),  # Use None if no scoring provided
                refit=optimize_for,
            )

            # Fit model
            fit_params = fit_params or {}
            estimators.fit(dataset.x_df(), dataset.y, **fit_params)

            # Get results for this optimization
            best_index = estimators.best_index_
            row_result = {
                "Optimized for": optimize_for,
                "Threshold": threshold,
                "Best Parameters": estimators.best_params_,
            }

            # Add scores for all metrics
            for metric in metrics:
                row_result[f"{metric}"] = estimators.cv_results_[f"mean_test_{metric}"][
                    best_index
                ]

            results.append(row_result)

    return results
