# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Shared preamble for the one-vs-rest multiclass classification tests.

ROCCurve, PrecisionRecallCurve, PopulationStabilityIndex and GINITable all need
the full per-class probability matrix for a multiclass model. The VMModel wrapper
only stores/exposes the positive-class column, so every one of those tests reaches
the underlying estimator's ``predict_proba``, guards the cases where it is missing,
shaped wrong, trained on fewer than 3 classes, or evaluated against labels the
model never saw (raising ``SkipTestError`` with a test-specific message), and
one-hot encodes the true labels. This module holds that preamble in one place.

Column alignment matters: scikit-learn orders ``predict_proba`` columns by
``estimator.classes_`` (the classes seen in training), which is not necessarily
``np.unique(dataset.y)`` — a dataset slice can be missing a training class. We
align on ``classes_`` when available (falling back to ``np.unique``) so the
probability columns line up, binarize against that full training class list to
keep the columns aligned, and expose the subset of classes actually present in the
evaluated dataset's ``y`` (a class with no positives has no defined ROC) so callers
emit per-class output only for those. Consequently, callers that compute a
micro-average pool only the present classes' aligned columns, so when a training
class is absent from the evaluated slice the micro number deliberately excludes
that class's column rather than pooling the full training-class matrix.
"""

from typing import Any, List, NamedTuple

import numpy as np
from sklearn.preprocessing import label_binarize

from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel


class MulticlassProba(NamedTuple):
    """Aligned per-class probabilities for a multiclass one-vs-rest test.

    ``class_list`` is the full training class order used for column alignment and
    binarization; ``classes_present``/``present_indices`` select the classes that
    actually appear in the evaluated dataset's ``y`` (paired: ``present_indices[k]``
    is the ``class_list``/``y_prob`` column for ``classes_present[k]``). ``y_bin`` is
    the true labels one-hot encoded against ``class_list``; ``y_prob`` is the
    estimator's per-class probability matrix in ``class_list`` column order.
    """

    class_list: np.ndarray
    classes_present: List[Any]
    present_indices: List[int]
    y_bin: np.ndarray
    y_prob: np.ndarray


def resolve_class_list(model: VMModel, dataset: VMDataset) -> np.ndarray:
    """Return the ``predict_proba`` column order for a multiclass model.

    scikit-learn orders columns by ``estimator.classes_`` (training classes), so
    prefer that; fall back to ``np.unique(dataset.y)`` for models that do not
    expose it.
    """
    raw_model = getattr(model, "model", None)
    classes = getattr(raw_model, "classes_", None)
    if classes is None:
        classes = np.unique(dataset.y)
    return np.asarray(classes)


def proba_matrix(
    model: VMModel, dataset: VMDataset, class_list: np.ndarray, test_name: str
) -> np.ndarray:
    """Return the estimator's per-class probability matrix, or ``SkipTestError``.

    The VMModel wrapper's ``predict_proba`` is binary-only (positive-class column
    only), so reach the underlying estimator for the full per-class matrix. Skips
    (rather than crashes) for models without a usable ``predict_proba``
    (metadata-only / precomputed single-column predictions) or a matrix whose width
    does not match ``class_list``.
    """
    raw_model = getattr(model, "model", None)
    proba_fn = getattr(raw_model, "predict_proba", None)
    if not callable(proba_fn):
        raise SkipTestError(
            f"Multiclass {test_name} requires per-class probabilities from the "
            "underlying model's predict_proba, which is not available for this "
            "model (e.g. metadata-only / precomputed predictions). Skipping."
        )
    try:
        y_prob = np.asarray(proba_fn(dataset.x_df()))
    except Exception as e:
        raise SkipTestError(
            f"Multiclass {test_name} could not compute per-class probabilities "
            f"({type(e).__name__}). Skipping."
        ) from e

    n_classes = len(class_list)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise SkipTestError(
            f"Multiclass {test_name} requires a per-class probability matrix with "
            f"one column per class (got shape {getattr(y_prob, 'shape', None)} for "
            f"{n_classes} classes). Skipping."
        )
    return y_prob


def multiclass_proba(
    model: VMModel, dataset: VMDataset, test_name: str
) -> MulticlassProba:
    """Resolve aligned per-class probabilities for a multiclass one-vs-rest test.

    Aligns on the estimator's training classes, binarizes the true labels against
    that full class list (so columns stay aligned even when the dataset is missing a
    class), and reports which classes are actually present in the evaluated ``y``.
    Raises ``SkipTestError`` for models that cannot supply a matching per-class
    probability matrix, that were trained on fewer than 3 classes (one-vs-rest
    multiclass metrics don't apply to a binary model), or when the evaluated
    dataset's ``y`` contains a label the model was never trained on.
    """
    class_list = resolve_class_list(model, dataset)
    if len(class_list) < 3:
        raise SkipTestError(
            f"Multiclass {test_name} requires a model trained on at least 3 "
            "classes for one-vs-rest multiclass metrics to apply, but this model "
            f"appears to be binary (trained on {len(class_list)} classes). "
            "Skipping."
        )

    y_prob = proba_matrix(model, dataset, class_list, test_name)

    y_true = np.asarray(dataset.y).flatten()
    y_bin = label_binarize(y_true, classes=class_list)

    unknown_labels = sorted(set(np.unique(y_true)) - set(class_list))
    if unknown_labels:
        raise SkipTestError(
            f"Multiclass {test_name} found label(s) {unknown_labels} in the "
            "evaluated dataset that the model was not trained on (not present in "
            "the estimator's classes_). Skipping."
        )

    present_indices = [i for i, cls in enumerate(class_list) if np.any(y_true == cls)]
    classes_present = [class_list[i] for i in present_indices]

    return MulticlassProba(
        class_list=class_list,
        classes_present=classes_present,
        present_indices=present_indices,
        y_bin=y_bin,
        y_prob=y_prob,
    )
