# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from validmind import tags, tasks
from validmind.tests.decorator import scorer
from validmind.vm_models import VMDataset


@scorer()
@tasks("classification")
@tags("classification", "outlier", "anomaly")
def OutlierScore(
    dataset: VMDataset, contamination: float = 0.1
) -> List[Dict[str, Any]]:
    """Calculates outlier scores and isolation paths for a classification model.

    Uses Isolation Forest to identify samples that deviate significantly from
    the typical patterns in the feature space. Returns both outlier scores and
    isolation paths, which provide insights into how anomalous each sample is
    and the path length through the isolation forest trees.

    Args:
        dataset: The dataset containing feature data
        contamination: Expected proportion of outliers, defaults to 0.1

    Returns:
        List[Dict[str, Any]]: Per-row outlier metrics as a list of dictionaries.
        Each dictionary contains:
        - "outlier_score": float - Normalized outlier score (0-1, higher = more outlier-like)
        - "isolation_path": float - Average path length through isolation forest trees
        - "anomaly_score": float - Raw anomaly score from isolation forest
        - "is_outlier": bool - Whether the sample is classified as an outlier

    Note:
        Outlier scores are normalized to [0, 1] where higher values indicate more outlier-like samples.
        Isolation paths represent the average number of splits required to isolate a sample.
    """
    # Get feature data
    X = dataset.x_df()

    # Handle case where we have no features or only categorical features
    if X.empty or X.shape[1] == 0:
        # Return zero outlier scores if no features available
        return [
            {
                "outlier_score": 0.0,
                "isolation_path": 0.0,
                "anomaly_score": 0.0,
                "is_outlier": False,
            }
        ] * len(dataset.y)

    # Select only numeric features for outlier detection
    numeric_features = dataset.feature_columns_numeric
    if not numeric_features:
        # If no numeric features, return zero outlier scores
        return [
            {
                "outlier_score": 0.0,
                "isolation_path": 0.0,
                "anomaly_score": 0.0,
                "is_outlier": False,
            }
        ] * len(dataset.y)

    X_numeric = X[numeric_features]

    # Handle missing values by filling with median
    X_filled = X_numeric.fillna(X_numeric.median())

    # Standardize features for better outlier detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Fit Isolation Forest
    isolation_forest = IsolationForest(
        contamination=contamination, random_state=42, n_estimators=100
    )

    # Fit the model on the data
    isolation_forest.fit(X_scaled)

    # Get anomaly scores (negative values for outliers)
    anomaly_scores = isolation_forest.decision_function(X_scaled)

    # Get outlier predictions (True for outliers)
    outlier_predictions = isolation_forest.predict(X_scaled) == -1

    # Calculate isolation paths (average path length through trees)
    isolation_paths = _calculate_isolation_paths(isolation_forest, X_scaled)

    # Convert to outlier scores (0 to 1, where 1 is most outlier-like)
    # Normalize using min-max scaling
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)

    if max_score == min_score:
        # All samples have same score, no outliers detected
        outlier_scores = np.zeros_like(anomaly_scores)
    else:
        # Invert and normalize: higher values = more outlier-like
        outlier_scores = (max_score - anomaly_scores) / (max_score - min_score)

    # Create list of dictionaries with all metrics
    results = []
    for i in range(len(outlier_scores)):
        results.append(
            {
                "outlier_score": float(outlier_scores[i]),
                "isolation_path": float(isolation_paths[i]),
                "anomaly_score": float(anomaly_scores[i]),
                "is_outlier": bool(outlier_predictions[i]),
            }
        )

    return results


def _calculate_isolation_paths(isolation_forest, X):
    """Calculate average isolation path lengths for each sample."""
    paths = []

    for sample in X:
        # Get path lengths from all trees
        sample_paths = []
        for tree in isolation_forest.estimators_:
            # Get the path length for this sample in this tree
            path_length = _get_path_length(tree, sample.reshape(1, -1))
            sample_paths.append(path_length)

        # Average path length across all trees
        avg_path_length = np.mean(sample_paths)
        paths.append(avg_path_length)

    return np.array(paths)


def _get_path_length(tree, X):
    """Get the path length for a sample in a single tree."""
    # This is a simplified version - in practice, you might want to use
    # the tree's decision_path method for more accurate path lengths
    try:
        # Use the tree's decision_path to get the path
        path = tree.decision_path(X)
        # Count the number of nodes in the path (excluding leaf)
        path_length = path.nnz - 1
        return path_length
    except Exception:
        # Fallback: estimate path length based on tree depth
        return tree.get_depth()
