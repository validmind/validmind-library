# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from sklearn.metrics import f1_score


def F1(model, dataset, **kwargs):
    return f1_score(dataset.y, dataset.y_pred(model), **kwargs)