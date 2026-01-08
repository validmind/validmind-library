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
def ClassBalance(model: VMModel, dataset: VMDataset) -> List[float]:
    """Calculates the class balance score per row for a classification model.

    For each prediction, this returns how balanced the predicted class is in the
    training distribution. Lower scores indicate predictions on rare classes,
    higher scores indicate predictions on common classes. This helps understand
    if model errors are more likely on imbalanced classes.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predictions

    Returns:
        List[float]: Per-row class balance scores as a list of float values

    Note:
        Scores range from 0 to 0.5, where 0.5 indicates perfectly balanced classes
        and lower values indicate more imbalanced classes.
    """
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate class frequencies in the true labels (proxy for training distribution)
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    class_frequencies = class_counts / len(y_true)

    # Create a mapping from class to frequency
    class_to_freq = dict(zip(unique_classes, class_frequencies))

    # Calculate balance score for each prediction
    balance_scores = []

    for pred in y_pred:
        if pred in class_to_freq:
            freq = class_to_freq[pred]
            # Balance score: how close to 0.5 (perfectly balanced) the frequency is
            # Score = 0.5 - |freq - 0.5| = min(freq, 1-freq)
            balance_score = min(freq, 1 - freq)
        else:
            # Predicted class not seen in true labels (very rare)
            balance_score = 0.0

        balance_scores.append(balance_score)

    # Return as a list of floats
    return balance_scores
