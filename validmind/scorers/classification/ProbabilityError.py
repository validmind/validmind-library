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
def ProbabilityError(model: VMModel, dataset: VMDataset) -> List[float]:
    """Calculates the probability error per row for a classification model.

    For binary classification tasks, this computes the absolute difference between
    the true class labels (0 or 1) and the predicted probabilities for each row.
    This provides insight into how confident the model's predictions are and
    how far off they are from the actual labels.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predicted probabilities

    Returns:
        List[float]: Per-row probability errors as a list of float values

    Raises:
        ValueError: If probability column is not found for the model
    """
    y_true = dataset.y

    # Try to get probabilities, fall back to predictions if not available
    try:
        y_prob = dataset.y_prob(model)
        # For binary classification, use the positive class probability
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]  # Use probability of positive class
    except ValueError:
        # Fall back to predictions if probabilities not available
        y_prob = dataset.y_pred(model)

    # Convert to numpy arrays and ensure same data type
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Compute absolute difference between true labels and predicted probabilities
    probability_errors = np.abs(y_true - y_prob)

    # Return as a list of floats
    return probability_errors.tolist()
