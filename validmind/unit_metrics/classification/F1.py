# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from sklearn.metrics import f1_score

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tasks("classification")
@tags("classification")
def F1(model: VMModel, dataset: VMDataset, **kwargs) -> float:
    """Calculates the F1 score for a classification model."""
    return f1_score(dataset.y, dataset.y_pred(model), **kwargs)
