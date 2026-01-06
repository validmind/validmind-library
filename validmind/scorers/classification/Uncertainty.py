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
def Uncertainty(model: VMModel, dataset: VMDataset) -> List[float]:
    """Calculates the prediction uncertainty per row for a classification model.

    Uncertainty is measured using the entropy of the predicted probability distribution.
    Higher entropy indicates higher uncertainty in the prediction. For binary
    classification, maximum uncertainty occurs at probability 0.5.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predicted probabilities

    Returns:
        List[float]: Per-row uncertainty scores as a list of float values

    Raises:
        ValueError: If probability column is not found for the model
    """
    # Try to get probabilities
    try:
        y_prob = dataset.y_prob(model)

        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            # Multi-class: calculate entropy across all classes
            # Clip to avoid log(0)
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            # Entropy: -sum(p * log(p))
            uncertainty = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)
        else:
            # Binary classification: calculate binary entropy
            y_prob = np.asarray(y_prob, dtype=float)
            # Clip to avoid log(0)
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
            uncertainty = -(
                y_prob_clipped * np.log(y_prob_clipped)
                + (1 - y_prob_clipped) * np.log(1 - y_prob_clipped)
            )

    except ValueError:
        # If no probabilities available, assume zero uncertainty for hard predictions
        n_samples = len(dataset.y)
        uncertainty = np.zeros(n_samples)

    # Return as a list of floats
    return uncertainty.tolist()
