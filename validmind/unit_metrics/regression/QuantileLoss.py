# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np

from validmind import tags, tasks
from validmind.vm_models.result.result import MetricValues


@tags("regression")
@tasks("regression")
def QuantileLoss(model, dataset, quantile=0.5) -> float:
    """Calculates the quantile loss for a regression model."""
    error = dataset.y - dataset.y_pred(model)

    return MetricValues(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))
