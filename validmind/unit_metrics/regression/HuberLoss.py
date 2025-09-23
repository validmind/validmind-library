# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("regression")
@tasks("regression")
def HuberLoss(model: VMModel, dataset: VMDataset) -> float:
    """Calculates the Huber loss for a regression model."""
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # delta - Threshold for the squared error to be linear or quadratic.
    delta = 1.0
    error = y_true - y_pred

    quadratic_part = np.minimum(np.abs(error), delta)
    linear_part = np.abs(error) - quadratic_part

    return np.mean(0.5 * quadratic_part**2 + delta * linear_part)
