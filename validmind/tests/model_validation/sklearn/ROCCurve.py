# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "model_performance",
    "visualization",
)
@tasks("classification", "text_classification")
def ROCCurve(model: VMModel, dataset: VMDataset) -> Tuple[go.Figure, RawData]:
    """
    Evaluates classification model performance by generating and plotting the Receiver Operating Characteristic
    (ROC) curve and calculating the Area Under Curve (AUC) score, for both binary and multiclass models.

    ### Purpose

    The Receiver Operating Characteristic (ROC) curve evaluates the performance of classification models. This curve
    illustrates the balance between the True Positive Rate (TPR) and False Positive Rate (FPR) across various threshold
    levels. In combination with the Area Under the Curve (AUC), the ROC curve measures the model's discrimination
    ability between classes. For binary problems (e.g., default vs non-default) a single curve is drawn for the
    positive class. For multiclass problems the curve is computed one-vs-rest — one curve and AUC per class, plus a
    micro-average across all classes — so the model's discrimination ability can be assessed for every class. Ideally,
    a higher AUC score signifies superior model performance in accurately distinguishing between classes.

    ### Test Mechanism

    This test selects the target model and dataset and determines the number of classes from the true labels. For
    binary targets it computes the predicted probabilities for the positive class and, together with the true
    outcomes, generates and plots a single ROC curve. For multiclass targets it obtains the full per-class probability
    matrix from the model and plots a one-vs-rest curve for each class along with a micro-average curve. In both cases
    a line signifying randomness (AUC of 0.5) is included, and the AUC score(s) are computed as a numerical estimation
    of performance. If any Infinite values are detected in the ROC threshold, these are effectively eliminated. The
    resulting ROC curves, AUC scores, and thresholds are consequently saved for future reference.

    ### Signs of High Risk

    - A high risk is potentially linked to the model's performance if the AUC score drops below or nears 0.5.
    - Another warning sign would be the ROC curve lying closer to the line of randomness, indicating no discriminative
    ability.
    - For the model to be deemed competent at its classification tasks, it is crucial that the AUC score is
    significantly above 0.5.

    ### Strengths

    - The ROC Curve offers an inclusive visual depiction of a model's discriminative power throughout all conceivable
    classification thresholds, unlike other metrics that solely disclose model performance at one fixed threshold.
    - Despite the proportions of the dataset, the AUC Score, which represents the entire ROC curve as a single data
    point, continues to be consistent, proving to be the ideal choice for such situations.

    ### Limitations

    - For multiclass models the curve is computed one-vs-rest (one curve per class plus a micro-average), which
    requires per-class probabilities from the model's `predict_proba`. Models that cannot produce a full per-class
    probability matrix (e.g. metadata-only models, or predictions supplied as a single precomputed probability column)
    are skipped for the multiclass case.
    - Furthermore, its performance might be subpar with models that output probabilities highly skewed towards 0 or 1.
    - At the extreme, the ROC curve could reflect high performance even when the majority of classifications are
    incorrect, provided that the model's ranking format is retained. This phenomenon is commonly termed the "Class
    Imbalance Problem".
    """
    classes = np.unique(dataset.y)

    if len(classes) > 2:
        return _multiclass_roc_curve(model, dataset, classes)

    y_prob = dataset.y_prob(model)
    y_true = dataset.y.astype(y_prob.dtype).flatten()

    fpr, tpr, _ = roc_curve(y_true, y_prob, drop_intermediate=False)
    auc = roc_auc_score(y_true, y_prob)

    return (
        go.Figure(
            data=[
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC curve (AUC = {auc:.2f})",
                    line=dict(color="#DE257E"),
                ),
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random (AUC = 0.5)",
                    line=dict(color="grey", dash="dash"),
                ),
            ],
            layout=go.Layout(
                title=f"ROC Curve for {model.input_id} on {dataset.input_id}",
                xaxis=dict(title="False Positive Rate"),
                yaxis=dict(title="True Positive Rate"),
                width=700,
                height=500,
            ),
        ),
        RawData(
            fpr=fpr, tpr=tpr, auc=auc, model=model.input_id, dataset=dataset.input_id
        ),
    )


def _multiclass_roc_curve(
    model: VMModel, dataset: VMDataset, classes: np.ndarray
) -> Tuple[go.Figure, RawData]:
    """One-vs-rest ROC curves for a multiclass model.

    Needs the full per-class probability matrix, which the stored single
    probability column cannot provide, so we ask the model for it directly.
    Models without a usable ``predict_proba`` (metadata-only, precomputed
    single-column probabilities) are skipped rather than crashed.
    """
    # The VMModel wrapper's predict_proba is binary-only (it returns just the
    # positive-class column), so reach the underlying estimator for the full
    # per-class probability matrix.
    raw_model = getattr(model, "model", None)
    proba_fn = getattr(raw_model, "predict_proba", None)
    if not callable(proba_fn):
        raise SkipTestError(
            "Multiclass ROC Curve requires per-class probabilities from the "
            "underlying model's predict_proba, which is not available for this "
            "model (e.g. metadata-only / precomputed predictions). Skipping."
        )
    try:
        y_prob = np.asarray(proba_fn(dataset.x_df()))
    except Exception as e:
        raise SkipTestError(
            "Multiclass ROC Curve could not compute per-class probabilities "
            f"({type(e).__name__}). Skipping."
        ) from e

    n_classes = len(classes)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise SkipTestError(
            "Multiclass ROC Curve requires a per-class probability matrix with "
            f"one column per class (got shape {getattr(y_prob, 'shape', None)} "
            f"for {n_classes} classes). Skipping."
        )

    # One-hot the true labels in the same class order predict_proba columns use
    # (sklearn orders predict_proba columns by sorted class label == np.unique).
    y_true = dataset.y.flatten()
    y_bin = label_binarize(y_true, classes=classes)

    traces = []
    raw_fpr = {}
    raw_tpr = {}
    raw_auc = {}
    palette = ["#DE257E", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i], drop_intermediate=False)
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        key = str(cls)
        raw_fpr[key] = fpr
        raw_tpr[key] = tpr
        raw_auc[key] = auc
        traces.append(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Class {key} (AUC = {auc:.2f})",
                line=dict(color=palette[i % len(palette)]),
            )
        )

    # Micro-average across all one-vs-rest decisions.
    micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    micro_auc = roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")
    raw_fpr["micro"] = micro_fpr
    raw_tpr["micro"] = micro_tpr
    raw_auc["micro"] = micro_auc
    traces.append(
        go.Scatter(
            x=micro_fpr,
            y=micro_tpr,
            mode="lines",
            name=f"Micro-average (AUC = {micro_auc:.2f})",
            line=dict(color="black", dash="dot"),
        )
    )

    traces.append(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random (AUC = 0.5)",
            line=dict(color="grey", dash="dash"),
        )
    )

    return (
        go.Figure(
            data=traces,
            layout=go.Layout(
                title=f"ROC Curve (one-vs-rest) for {model.input_id} on {dataset.input_id}",
                xaxis=dict(title="False Positive Rate"),
                yaxis=dict(title="True Positive Rate"),
                width=700,
                height=500,
            ),
        ),
        RawData(
            fpr=raw_fpr,
            tpr=raw_tpr,
            auc=raw_auc,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
