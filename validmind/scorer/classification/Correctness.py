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
def Correctness(model: VMModel, dataset: VMDataset) -> List[int]:
    """Calculates the correctness per row for a classification model.

    For classification tasks, this returns 1 for correctly classified rows
    and 0 for incorrectly classified rows. This provides a binary indicator
    of model performance for each individual prediction.

    Args:
        model: The classification model to evaluate
        dataset: The dataset containing true labels and predictions

    Returns:
        List[int]: Per-row correctness as a list of 1s and 0s
    """
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # For classification, check if predictions match true labels
    correctness = (y_true == y_pred).astype(int)

    # Return as a list of integers
    return correctness.tolist()
