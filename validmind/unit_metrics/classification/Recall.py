# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from sklearn.metrics import recall_score

from validmind import tags, tasks


@tasks("classification")
@tags("classification")
def Recall(model, dataset, **kwargs):
    """Calculates the recall for a classification model."""
    return recall_score(dataset.y, dataset.y_pred(model), **kwargs)
