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
def CalibrationError(
    model: VMModel, dataset: VMDataset, n_bins: int = 10
) -> List[float]:
    """Calculates the calibration error per row for a classification model.

    Calibration error measures how well the predicted probabilities reflect the
    actual likelihood of the positive class. For each prediction, this computes
    the absolute difference between the predicted probability and the empirical
    frequency of the positive class in the corresponding probability bin.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predicted probabilities
        n_bins: Number of bins for probability calibration, defaults to 10

    Returns:
        List[float]: Per-row calibration errors as a list of float values

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
        # If no probabilities available, return zeros (perfect calibration for hard predictions)
        return [0.0] * len(y_true)

    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Create probability bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Calculate calibration error for each sample
    calibration_errors = np.zeros_like(y_prob)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if not np.any(in_bin):
            continue

        # Calculate empirical frequency for this bin
        empirical_freq = np.mean(y_true[in_bin])

        # Calculate average predicted probability for this bin
        avg_predicted_prob = np.mean(y_prob[in_bin])

        # Assign calibration error to all samples in this bin
        calibration_errors[in_bin] = abs(avg_predicted_prob - empirical_freq)

    # Return as a list of floats
    return calibration_errors.tolist()
