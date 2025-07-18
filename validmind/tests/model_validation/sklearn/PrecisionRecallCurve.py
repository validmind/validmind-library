# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.models import FoundationModel
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "binary_classification", "model_performance", "visualization")
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

    - This metric is only applicable to binary classification models - it raises errors for multiclass classification
    models or Foundation models.
    - It may not fully represent the overall accuracy of the model if the cost of false positives and false negatives
    are extremely different, or if the dataset is heavily imbalanced.
    """
    if isinstance(model, FoundationModel):
        raise SkipTestError("Skipping PrecisionRecallCurve for Foundation models")

    y_true = dataset.y
    if len(np.unique(y_true)) > 2:
        raise SkipTestError(
            "Precision Recall Curve is only supported for binary classification models"
        )

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
