# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from numpy import unique
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tasks("classification")
@tags("classification")
def ROC_AUC(model: VMModel, dataset: VMDataset, **kwargs) -> float:
    """Calculates the ROC AUC for a classification model."""
    y_true = dataset.y

    if len(unique(y_true)) > 2:
        y_pred = dataset.y_pred(model)
        y_true = y_true.astype(y_pred.dtype)
        roc_auc = _multiclass_roc_auc_score(y_true, y_pred, **kwargs)
    else:
        y_prob = dataset.y_prob(model)
        y_true = y_true.astype(y_prob.dtype).flatten()
        roc_auc = roc_auc_score(y_true, y_prob, **kwargs)

    return roc_auc


def _multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)

    return roc_auc_score(lb.transform(y_test), lb.transform(y_pred), average=average)
