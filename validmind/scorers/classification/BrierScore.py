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
def BrierScore(model: VMModel, dataset: VMDataset) -> List[float]:
    """Calculates the Brier score per row for a classification model.

    The Brier score is a proper score function that measures the accuracy of
    probabilistic predictions. It is calculated as the mean squared difference
    between predicted probabilities and the actual binary outcomes.
    Lower scores indicate better calibration.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predicted probabilities

    Returns:
        List[float]: Per-row Brier scores as a list of float values

    Raises:
        ValueError: If probability column is not found for the model
    """
    y_true = dataset.y

    # Try to get probabilities
    try:
        y_prob = dataset.y_prob(model)
        # For binary classification, use the positive class probability
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]  # Use probability of positive class
    except ValueError:
        # Fall back to predictions if probabilities not available
        # Convert predictions to "probabilities" (1.0 for predicted class, 0.0 for other)
        y_pred = dataset.y_pred(model)
        y_prob = y_pred.astype(float)

    # Convert to numpy arrays and ensure same data type
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Calculate Brier score per row: (predicted_probability - actual_outcome)²
    brier_scores = (y_prob - y_true) ** 2

    # Return as a list of floats
    return brier_scores.tolist()
