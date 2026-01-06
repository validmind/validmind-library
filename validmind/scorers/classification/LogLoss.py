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
def LogLoss(model: VMModel, dataset: VMDataset, eps: float = 1e-15) -> List[float]:
    """Calculates the logarithmic loss per row for a classification model.

    Log loss measures the performance of a classification model where the prediction
    is a probability value between 0 and 1. The log loss increases as the predicted
    probability diverges from the actual label.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predicted probabilities
        eps: Small value to avoid log(0), defaults to 1e-15

    Returns:
        List[float]: Per-row log loss values as a list of float values

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
        # Convert predictions to "probabilities" (0.99 for correct class, 0.01 for wrong)
        y_pred = dataset.y_pred(model)
        y_prob = np.where(y_true == y_pred, 0.99, 0.01)

    # Convert to numpy arrays and ensure same data type
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Clip probabilities to avoid log(0) and log(1)
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # Calculate log loss per row: -[y*log(p) + (1-y)*log(1-p)]
    log_loss_per_row = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    # Return as a list of floats
    return log_loss_per_row.tolist()
