# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from sklearn.metrics import r2_score

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("regression")
@tasks("regression")
def RSquaredScore(model: VMModel, dataset: VMDataset) -> float:
    """Calculates the R-squared score for a regression model."""
    return r2_score(dataset.y, dataset.y_pred(model))
