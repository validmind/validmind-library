# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.models import FoundationModel
from validmind.vm_models import VMDataset, VMModel

from ._multiclass_proba import multiclass_proba


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "model_performance",
    "visualization",
)
@tasks("classification", "text_classification")
def PrecisionRecallCurve(
    model: VMModel, dataset: VMDataset
) -> Tuple[go.Figure, RawData]:
    """
    Evaluates the precision-recall trade-off for binary classification models and visualizes the Precision-Recall curve.

    ### Purpose

    The Precision Recall Curve metric is intended to evaluate the trade-off between precision and recall in
    classification models, particularly binary classification models. It assesses the model's capacity to produce
    accurate results (high precision), as well as its ability to capture a majority of all positive instances (high
    recall).

    ### Test Mechanism

    The test extracts ground truth labels and prediction probabilities from the model's test dataset. It applies the
    `precision_recall_curve` method from the sklearn metrics module to these extracted labels and predictions, which
    computes a precision-recall pair for each possible threshold. This calculation results in an array of precision and
    recall scores that can be plotted against each other to form the Precision-Recall Curve. This curve is then
    visually represented by using Plotly's scatter plot.

    ### Signs of High Risk

    - A lower area under the Precision-Recall Curve signifies high risk.
    - This corresponds to a model yielding a high amount of false positives (low precision) and/or false negatives (low
    recall).
    - If the curve is closer to the bottom left of the plot, rather than being closer to the top right corner, it can
    be a sign of high risk.

    ### Strengths

    - This metric aptly represents the balance between precision (minimizing false positives) and recall (minimizing
    false negatives), which is especially critical in scenarios where both values are significant.
    - Through the graphic representation, it enables an intuitive understanding of the model's performance across
    different threshold levels.

    ### Limitations

    - For multiclass models the curve is computed one-vs-rest (one curve per class plus a micro-average), which
    requires per-class probabilities from the model's `predict_proba`. Models that cannot produce a full per-class
    probability matrix (e.g. Foundation/metadata-only models, or predictions supplied as a single precomputed
    probability column) are skipped for the multiclass case.
    - It may not fully represent the overall accuracy of the model if the cost of false positives and false negatives
    are extremely different, or if the dataset is heavily imbalanced.
    """
    if isinstance(model, FoundationModel):
        raise SkipTestError("Skipping PrecisionRecallCurve for Foundation models")

    y_true = dataset.y
    classes = np.unique(y_true)

    if len(classes) > 2:
        return _multiclass_pr_curve(model, dataset)

    precision, recall, _ = precision_recall_curve(y_true, dataset.y_prob(model))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name="Precision-Recall Curve",
                line=dict(color="#DE257E"),
            )
        ],
        layout=go.Layout(
            title="Precision-Recall Curve",
            xaxis=dict(title="Recall"),
            yaxis=dict(title="Precision"),
        ),
    )

    return fig, RawData(
        precision=precision,
        recall=recall,
        model=model.input_id,
        dataset=dataset.input_id,
    )


def _multiclass_pr_curve(
    model: VMModel, dataset: VMDataset
) -> Tuple[go.Figure, RawData]:
    """One-vs-rest precision-recall curves for a multiclass model.

    Needs the full per-class probability matrix, which the stored single
    probability column cannot provide; the shared helper reaches the underlying
    estimator, aligns the probability columns to the training class order and
    skips models that cannot supply a matching matrix.
    """
    aligned = multiclass_proba(model, dataset, "Precision-Recall Curve")
    y_bin = aligned.y_bin
    y_prob = aligned.y_prob

    traces = []
    raw_precision = {}
    raw_recall = {}
    raw_ap = {}
    palette = ["#DE257E", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]
    for plot_i, (i, cls) in enumerate(
        zip(aligned.present_indices, aligned.classes_present)
    ):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        key = str(cls)
        raw_precision[key] = precision
        raw_recall[key] = recall
        raw_ap[key] = ap
        traces.append(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"Class {key} (AP = {ap:.2f})",
                line=dict(color=palette[plot_i % len(palette)]),
            )
        )

    # Micro-average across the one-vs-rest decisions of the present classes.
    present = aligned.present_indices
    y_bin = y_bin[:, present]
    y_prob = y_prob[:, present]
    micro_precision, micro_recall, _ = precision_recall_curve(
        y_bin.ravel(), y_prob.ravel()
    )
    micro_ap = average_precision_score(y_bin, y_prob, average="micro")
    raw_precision["micro"] = micro_precision
    raw_recall["micro"] = micro_recall
    raw_ap["micro"] = micro_ap
    traces.append(
        go.Scatter(
            x=micro_recall,
            y=micro_precision,
            mode="lines",
            name=f"Micro-average (AP = {micro_ap:.2f})",
            line=dict(color="black", dash="dot"),
        )
    )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"Precision-Recall Curve (one-vs-rest) for {model.input_id} on {dataset.input_id}",
            xaxis=dict(title="Recall"),
            yaxis=dict(title="Precision"),
        ),
    )

    return fig, RawData(
        precision=raw_precision,
        recall=raw_recall,
        average_precision=raw_ap,
        model=model.input_id,
        dataset=dataset.input_id,
    )
