# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

# NumPy 2 removed `np.trapz` in favour of `np.trapezoid`; the library supports both
# (numpy<2 below Python 3.14, numpy>=2.3 from 3.14 on).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@tags("regression")
@tasks("regression")
def GiniCoefficient(dataset: VMDataset, model: VMModel) -> float:
    """Calculates the Gini coefficient for a regression model."""
    y_true = dataset.y
    y_pred = dataset.y_pred(model)

    # Sort true values and corresponding predicted values
    idx = np.argsort(y_true)
    y_true_sorted = y_true[idx]
    y_pred_sorted = y_pred[idx]

    # Compute cumulative sums
    cumsum_true = np.cumsum(y_true_sorted)
    cumsum_pred = np.cumsum(y_pred_sorted)

    # Normalize cumulative sums
    cumsum_true_norm = cumsum_true / np.max(cumsum_true)
    cumsum_pred_norm = cumsum_pred / np.max(cumsum_pred)

    # Compute area under the Lorenz curve
    area_lorenz = _trapezoid(cumsum_pred_norm, x=cumsum_true_norm)

    # Compute Gini coefficient
    return 1 - 2 * area_lorenz
