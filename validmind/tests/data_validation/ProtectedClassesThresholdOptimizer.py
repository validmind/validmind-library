# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
import json
import matplotlib.pyplot as plt
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_ratio,
    equalized_odds_ratio,
)
from fairlearn.metrics import (
    count,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
)


def ProtectedClassesThresholdOptimizer(
    dataset, pipeline, protected_classes, X_train, y_train
):
    """
    Obtains a classifier by applying group-specific thresholds to the provided estimator.

    ### Purpose

    This test aims to optimize the fairness of a machine learning model by applying different
    classification thresholds for different protected groups. It helps in mitigating bias and
    achieving more equitable outcomes across different demographic groups.

    ### Test Mechanism

    The test uses Fairlearn's ThresholdOptimizer to:
    1. Fit an optimizer on the training data, considering protected classes.
    2. Apply optimized thresholds to make predictions on the test data.
    3. Calculate and report various fairness metrics.
    4. Visualize the optimized thresholds.

    ### Signs of High Risk

    - Large disparities in fairness metrics (e.g., Demographic Parity Ratio, Equalized Odds Ratio)
      across different protected groups.
    - Significant differences in False Positive Rates (FPR) or True Positive Rates (TPR) between groups.
    - Thresholds that vary widely across different protected groups.

    ### Strengths

    - Provides a post-processing method to improve model fairness without modifying the original model.
    - Allows for balancing multiple fairness criteria simultaneously.
    - Offers visual insights into the threshold optimization process.

    ### Limitations

    - May lead to a decrease in overall model performance while improving fairness.
    - Requires access to protected attribute information at prediction time.
    - The effectiveness can vary depending on the chosen fairness constraint and objective.
    """
    test_df = dataset.df

    threshold_optimizer = initialize_and_fit_optimizer(
        pipeline, X_train, y_train, X_train[protected_classes]
    )

    fig = plot_thresholds(threshold_optimizer)

    target = dataset.target_column
    y_pred_opt = make_predictions(threshold_optimizer, test_df, protected_classes)

    fairness_metrics = calculate_fairness_metrics(
        test_df, target, y_pred_opt, protected_classes
    )

    return (
        {"DPR and EOR Table": fairness_metrics.reset_index()},
        fig,
    )


def initialize_and_fit_optimizer(pipeline, X_train, y_train, protected_classes_df):
    threshold_optimizer = ThresholdOptimizer(
        estimator=pipeline,
        objective="balanced_accuracy_score",
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=False,
    )
    threshold_optimizer.fit(X_train, y_train, sensitive_features=protected_classes_df)
    return threshold_optimizer


def plot_thresholds(threshold_optimizer):
    fig = plt.figure()
    plot_threshold_optimizer(threshold_optimizer, show_plot=False)
    return fig


def make_predictions(threshold_optimizer, test_df, protected_classes):
    y_pred_opt = threshold_optimizer.predict(
        test_df, sensitive_features=test_df[protected_classes]
    )
    return y_pred_opt


def calculate_fairness_metrics(test_df, target, y_pred_opt, protected_classes):
    fairness_metrics = pd.DataFrame(
        columns=protected_classes,
        index=["demographic parity ratio", "equal odds ratio"],
    )

    for feature in protected_classes:
        dpr = demographic_parity_ratio(
            y_true=test_df[target],
            y_pred=y_pred_opt,
            sensitive_features=test_df[[feature]],
        )
        eor = equalized_odds_ratio(
            y_true=test_df[target],
            y_pred=y_pred_opt,
            sensitive_features=test_df[[feature]],
        )
        fairness_metrics[feature] = [round(dpr, 2), round(eor, 2)]

    return fairness_metrics


def calculate_group_metrics(test_df, target, y_pred_opt, protected_classes):
    metrics = {
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "fnr": false_negative_rate,
        "count": count,
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=test_df[target],
        y_pred=y_pred_opt,
        sensitive_features=test_df[protected_classes],
    )
    group_metrics = mf.by_group
    return group_metrics


def get_thresholds_by_group(threshold_optimizer):
    threshold_rules = threshold_optimizer.interpolated_thresholder_.interpolation_dict
    thresholds = json.dumps(threshold_rules, default=str, indent=4)
    thresholds_df = pd.DataFrame.from_records(json.loads(thresholds))
    return thresholds_df