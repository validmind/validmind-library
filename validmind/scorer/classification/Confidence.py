# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List

import numpy as np

from validmind import tags, tasks
from validmind.tests.decorator import scorer
from validmind.vm_models import VMDataset, VMModel


@scorer()
@tasks("classification")
@tags("classification")
def Confidence(model: VMModel, dataset: VMDataset) -> List[float]:
    """Calculates the prediction confidence per row for a classification model.

    For binary classification, confidence is calculated as the maximum probability
    across classes, or alternatively as the distance from the decision boundary (0.5).
    Higher values indicate more confident predictions.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predicted probabilities

    Returns:
        List[float]: Per-row confidence scores as a list of float values

    Raises:
        ValueError: If probability column is not found for the model
    """
    # Try to get probabilities, fall back to predictions if not available
    try:
        y_prob = dataset.y_prob(model)
        # For binary classification, use max probability approach
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            # Multi-class: confidence is the maximum probability
            confidence = np.max(y_prob, axis=1)
        else:
            # Binary classification: confidence based on distance from 0.5
            y_prob = np.asarray(y_prob, dtype=float)
            confidence = np.abs(y_prob - 0.5) + 0.5
    except ValueError:
        # Fall back to binary correctness if probabilities not available
        y_true = dataset.y
        y_pred = dataset.y_pred(model)
        # If no probabilities, confidence is 1.0 for correct, 0.0 for incorrect
        confidence = (y_true == y_pred).astype(float)

    # Return as a list of floats
    return confidence.tolist()
