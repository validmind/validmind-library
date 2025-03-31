# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
    target_recall: Optional[float] = None,
) -> Dict[str, Union[str, float]]:
    """
    Find the optimal classification threshold using various methods.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        method: Method to use for finding optimal threshold
        target_recall: Required if method='target_recall'

    Returns:
        dict: Dictionary containing threshold and metrics
    """
    # Get ROC and PR curve points
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)

    # Find optimal threshold based on method
    if method == "naive":
        optimal_threshold = 0.5
    elif method == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds_roc[best_idx]
    elif method == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 1.0
        )
    elif method == "precision_recall":
        diff = abs(precision - recall)
        best_idx = np.argmin(diff)
        optimal_threshold = (
            thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 1.0
        )
    elif method == "target_recall":
        if target_recall is None:
            raise ValueError(
                "target_recall must be specified when method='target_recall'"
            )
        idx = np.argmin(abs(recall - target_recall))
        optimal_threshold = thresholds_pr[idx] if idx < len(thresholds_pr) else 1.0
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate predictions with optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics directly
    metrics = {
        "method": method,
        "threshold": optimal_threshold,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }

    return metrics


@tags("model_validation", "threshold_optimization", "classification_metrics")
@tasks("classification")
def ClassifierThresholdOptimization(
    dataset: VMDataset,
    model: VMModel,
    methods: Optional[List[str]] = None,
    target_recall: Optional[float] = None,
) -> Dict[str, Union[pd.DataFrame, go.Figure]]:
    """
    Analyzes and visualizes different threshold optimization methods for binary classification models.

    ### Purpose

    The Classifier Threshold Optimization test identifies optimal decision thresholds using various
    methods to balance different performance metrics. This helps adapt the model's decision boundary
    to specific business requirements, such as minimizing false positives in fraud detection or
    achieving target recall in medical diagnosis.

    ### Test Mechanism

    The test implements multiple threshold optimization methods:
    1. Youden's J statistic (maximizing sensitivity + specificity - 1)
    2. F1-score optimization (balancing precision and recall)
    3. Precision-Recall equality point
    4. Target recall achievement
    5. Naive (0.5) threshold
    For each method, it computes ROC and PR curves, identifies optimal points, and provides
    comprehensive performance metrics at each threshold.

    ### Signs of High Risk

    - Large discrepancies between different optimization methods
    - Optimal thresholds far from the default 0.5
    - Poor performance metrics across all thresholds
    - Significant gap between achieved and target recall
    - Unstable thresholds across different methods
    - Extreme trade-offs between precision and recall
    - Threshold optimization showing minimal impact
    - Business metrics not improving with optimization

    ### Strengths

    - Multiple optimization strategies for different needs
    - Visual and numerical results for comparison
    - Support for business-driven optimization (target recall)
    - Comprehensive performance metrics at each threshold
    - Integration with ROC and PR curves
    - Handles class imbalance through various metrics
    - Enables informed threshold selection
    - Supports cost-sensitive decision making

    ### Limitations

    - Assumes cost of false positives/negatives are known
    - May need adjustment for highly imbalanced datasets
    - Threshold might not be stable across different samples
    - Cannot handle multi-class problems directly
    - Optimization methods may conflict with business needs
    - Requires sufficient validation data
    - May not capture temporal changes in optimal threshold
    - Single threshold may not be optimal for all subgroups

    Args:
        dataset: VMDataset containing features and target
        model: VMModel containing predictions
        methods: List of methods to compare (default: ['youden', 'f1', 'precision_recall'])
        target_recall: Target recall value if using 'target_recall' method

    Returns:
        Dictionary containing:
            - table: DataFrame comparing different threshold optimization methods
                    (using weighted averages for precision, recall, and f1)
            - figure: Plotly figure showing ROC and PR curves with optimal thresholds
    """
    # Verify binary classification
    unique_values = np.unique(dataset.y)
    if len(unique_values) != 2:
        raise ValueError("Target variable must be binary")

    if methods is None:
        methods = ["naive", "youden", "f1", "precision_recall"]
        if target_recall is not None:
            methods.append("target_recall")

    y_true = dataset.y
    y_prob = dataset.y_prob(model)

    # Get curve points for plotting
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)

    # Calculate optimal thresholds and metrics
    results = []
    optimal_points = {}

    for method in methods:
        metrics = find_optimal_threshold(y_true, y_prob, method, target_recall)
        results.append(metrics)

        # Store optimal points for plotting
        if method == "youden":
            idx = np.argmax(tpr - fpr)
            optimal_points[method] = {
                "x": fpr[idx],
                "y": tpr[idx],
                "threshold": thresholds_roc[idx],
            }
        elif method in ["f1", "precision_recall", "target_recall"]:
            idx = np.argmin(abs(thresholds_pr - metrics["threshold"]))
            optimal_points[method] = {
                "x": recall[idx],
                "y": precision[idx],
                "threshold": metrics["threshold"],
            }

    # Create visualization
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve")
    )

    # Plot ROC curve
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name="ROC Curve", line=dict(color="blue")),
        row=1,
        col=1,
    )

    # Plot PR curve
    fig.add_trace(
        go.Scatter(x=recall, y=precision, name="PR Curve", line=dict(color="green")),
        row=1,
        col=2,
    )

    # Add optimal points
    colors = {
        "youden": "red",
        "f1": "orange",
        "precision_recall": "purple",
        "target_recall": "brown",
    }

    for method, points in optimal_points.items():
        if method == "youden":
            fig.add_trace(
                go.Scatter(
                    x=[points["x"]],
                    y=[points["y"]],
                    name=f'{method} (t={points["threshold"]:.2f})',
                    mode="markers",
                    marker=dict(size=10, color=colors[method]),
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[points["x"]],
                    y=[points["y"]],
                    name=f'{method} (t={points["threshold"]:.2f})',
                    mode="markers",
                    marker=dict(size=10, color=colors[method]),
                ),
                row=1,
                col=2,
            )

    # Update layout
    fig.update_layout(
        height=500, title_text="Threshold Optimization Analysis", showlegend=True
    )

    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    # Create results table and sort by threshold descending
    table = pd.DataFrame(results).sort_values("threshold", ascending=False)

    return (
        fig,
        table,
        RawData(
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            thresholds_roc=thresholds_roc,
            thresholds_pr=thresholds_pr,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
