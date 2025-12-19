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
def AbsoluteError(model: VMModel, dataset: VMDataset) -> List[float]:
    """Calculates the absolute error per row for a classification model.

    For classification tasks, this computes the absolute difference between
    the true class labels and predicted class labels for each individual row.
    For binary classification with probabilities, it can also compute the
    absolute difference between true labels and predicted probabilities.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predictions

    Returns:
        List[float]: Per-row absolute errors as a list of float values
    """
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Convert to numpy arrays and ensure same data type
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # For classification, compute absolute difference between true and predicted labels
    absolute_errors = np.abs(y_true - y_pred)

    # Return as a list of floats
    return absolute_errors.astype(float).tolist()
