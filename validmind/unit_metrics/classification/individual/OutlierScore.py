# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tasks("classification")
@tags("classification")
def OutlierScore(
    model: VMModel, dataset: VMDataset, contamination: float = 0.1, **kwargs
) -> List[float]:
    """Calculates the outlier score per row for a classification model.

    Uses Isolation Forest to identify samples that deviate significantly from
    the typical patterns in the feature space. Higher scores indicate more
    anomalous/outlier-like samples. This can help identify out-of-distribution
    samples or data points that might be harder to predict accurately.

    Args:
        model: The classification model to evaluate (unused but kept for consistency)
        dataset: The dataset containing feature data
        contamination: Expected proportion of outliers, defaults to 0.1
        **kwargs: Additional parameters (unused for compatibility)

    Returns:
        List[float]: Per-row outlier scores as a list of float values

    Note:
        Scores are normalized to [0, 1] where higher values indicate more outlier-like samples
    """
    # Get feature data
    X = dataset.x_df()

    # Handle case where we have no features or only categorical features
    if X.empty or X.shape[1] == 0:
        # Return zero outlier scores if no features available
        return [0.0] * len(dataset.y)

    # Select only numeric features for outlier detection
    numeric_features = dataset.feature_columns_numeric
    if not numeric_features:
        # If no numeric features, return zero outlier scores
        return [0.0] * len(dataset.y)

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

    # Return as a list of floats
    return outlier_scores.tolist()
