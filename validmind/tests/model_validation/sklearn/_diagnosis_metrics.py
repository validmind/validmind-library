# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Shared metric helpers for the per-region classification diagnosis tests.

WeakspotsDiagnosis, OverfitDiagnosis and RobustnessDiagnosis all score
classification metrics (precision/recall/F1/AUC) on subsets of the data -- feature
bins or noise-perturbed copies. scikit-learn's defaults break on those subsets:

- precision/recall/F1 default to ``average="binary"`` with ``pos_label=1``, which
  raises for multiclass targets ("Target is multiclass but average='binary'") and
  for binary targets encoded outside ``{0, 1}`` ("pos_label=1 is not a valid
  label");
- ``roc_auc_score`` defaults to ``multi_class="raise"`` and needs a per-class
  probability matrix, which the library does not retain for multiclass models
  (only a single probability column is stored).

The averaging strategy is resolved once from the full label set so every subset is
scored consistently, and multiclass AUC falls back to the label-binarize
convention already used by ``ClassifierPerformance``.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from validmind.vm_models import VMDataset, VMModel


def full_labels(datasets: List[VMDataset], model: VMModel) -> List[Any]:
    """Return the sorted union of true and predicted labels across every dataset.

    Predictions are included because scikit-learn derives a metric's target type
    from the union of ``y_true`` and ``y_pred``: a subset whose true labels
    collapse to two classes is still multiclass if the model predicts a third.
    """
    labels = set()
    for dataset in datasets:
        labels.update(np.unique(dataset.y).tolist())
        labels.update(np.unique(dataset.y_pred(model)).tolist())
    return sorted(labels)


def resolve_averaging(
    datasets: List[VMDataset], model: VMModel
) -> Tuple[str, Optional[Any]]:
    """Return the ``(average, pos_label)`` for precision/recall/F1.

    ``("macro", None)`` for multiclass; ``("binary", <largest label>)`` otherwise,
    which leaves the standard ``{0, 1}`` case on ``pos_label=1`` so existing binary
    results are unchanged.
    """
    labels = full_labels(datasets, model)
    if len(labels) > 2:
        return "macro", None
    return "binary", labels[-1]


def bind_averaging(
    metric_fn: Callable, average: Optional[str], pos_label: Optional[Any]
) -> Callable:
    """Bind averaging options onto a metric that accepts them; pass others through.

    accuracy, regression metrics and any callable without an ``average`` parameter
    are returned unchanged. ``pos_label`` is only bound for binary averaging, and
    ``zero_division=0`` when supported so empty or single-class subsets score 0
    instead of emitting scikit-learn warnings. ``average=None`` (regression) is a
    no-op.
    """
    if average is None:
        return metric_fn

    params = inspect.signature(metric_fn).parameters
    kwargs = {}
    if "average" in params:
        kwargs["average"] = average
        if pos_label is not None and "pos_label" in params:
            kwargs["pos_label"] = pos_label
        if "zero_division" in params:
            kwargs["zero_division"] = 0
    return functools.partial(metric_fn, **kwargs) if kwargs else metric_fn


def apply_averaging(
    metrics: Dict[str, Callable], average: Optional[str], pos_label: Optional[Any]
) -> Dict[str, Callable]:
    """Bind averaging options onto every metric in a ``name -> callable`` mapping."""
    return {
        name: bind_averaging(metric_fn, average, pos_label)
        for name, metric_fn in metrics.items()
    }


def multiclass_auc(y_true, y_pred, labels: List[Any]) -> float:
    """Compute ROC AUC for multiclass targets from predicted labels.

    The library retains only a single probability column, so a probability-based
    multiclass ROC AUC is not possible; this mirrors ``ClassifierPerformance`` by
    binarizing labels instead. Binarization is fit on the full label set so columns
    stay aligned across subsets that omit some classes; only classes actually
    present in ``y_true`` (with both positive and negative examples) are scored,
    then macro-averaged. Returns ``0.0`` when no class is scorable.
    """
    binarizer = LabelBinarizer().fit(np.asarray(labels))
    true_bin = binarizer.transform(np.asarray(y_true))
    pred_bin = binarizer.transform(np.asarray(y_pred))

    # LabelBinarizer collapses to a single column only for a two-class fit; the
    # diagnosis tests call this for multiclass, but guard defensively.
    if true_bin.shape[1] == 1:
        true_bin = np.column_stack([1 - true_bin, true_bin])
        pred_bin = np.column_stack([1 - pred_bin, pred_bin])

    n_rows = len(y_true)
    scorable = [
        i for i in range(true_bin.shape[1]) if 0 < int(true_bin[:, i].sum()) < n_rows
    ]
    if not scorable:
        return 0.0

    return roc_auc_score(true_bin[:, scorable], pred_bin[:, scorable], average="macro")
