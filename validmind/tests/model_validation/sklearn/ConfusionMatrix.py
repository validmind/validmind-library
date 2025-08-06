# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn",
    "binary_classification",
    "multiclass_classification",
    "model_performance",
    "visualization",
)
@tasks("classification", "text_classification")
def ConfusionMatrix(
    dataset: VMDataset,
    model: VMModel,
    threshold: float = 0.5,
) -> Tuple[go.Figure, RawData]:
    """
    Evaluates and visually represents the classification ML model's predictive performance using a Confusion Matrix
    heatmap.

    ### Purpose

    The Confusion Matrix tester is designed to assess the performance of a classification Machine Learning model. This
    performance is evaluated based on how well the model is able to correctly classify True Positives, True Negatives,
    False Positives, and False Negatives - fundamental aspects of model accuracy.

    ### Test Mechanism

    The mechanism used involves taking the predicted results (`y_test_predict`) from the classification model and
    comparing them against the actual values (`y_test_true`). A confusion matrix is built using the unique labels
    extracted from `y_test_true`, employing scikit-learn's metrics. The matrix is then visually rendered with the help
    of Plotly's `create_annotated_heatmap` function. A heatmap is created which provides a two-dimensional graphical
    representation of the model's performance, showcasing distributions of True Positives (TP), True Negatives (TN),
    False Positives (FP), and False Negatives (FN).

    ### Signs of High Risk

    - High numbers of False Positives (FP) and False Negatives (FN), depicting that the model is not effectively
    classifying the values.
    - Low numbers of True Positives (TP) and True Negatives (TN), implying that the model is struggling with correctly
    identifying class labels.

    ### Strengths

    - It provides a simplified yet comprehensive visual snapshot of the classification model's predictive performance.
    - It distinctly brings out True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives
    (FN), thus making it easier to focus on potential areas of improvement.
    - The matrix is beneficial in dealing with multi-class classification problems as it can provide a simple view of
    complex model performances.
    - It aids in understanding the different types of errors that the model could potentially make, as it provides
    in-depth insights into Type-I and Type-II errors.

    ### Limitations

    - In cases of unbalanced classes, the effectiveness of the confusion matrix might be lessened. It may wrongly
    interpret the accuracy of a model that is essentially just predicting the majority class.
    - It does not provide a single unified statistic that could evaluate the overall performance of the model.
    Different aspects of the model's performance are evaluated separately instead.
    - It mainly serves as a descriptive tool and does not offer the capability for statistical hypothesis testing.
    - Risks of misinterpretation exist because the matrix doesn't directly provide precision, recall, or F1-score data.
    These metrics have to be computed separately.
    """
    # Get predictions using threshold for binary classification if possible
    if hasattr(model.model, "predict_proba"):
        y_prob = dataset.y_prob(model)
        # Handle both 1D and 2D probability arrays
        if y_prob.ndim == 2:
            y_pred = (y_prob[:, 1] > threshold).astype(int)
        else:
            y_pred = (y_prob > threshold).astype(int)
    else:
        y_pred = dataset.y_pred(model)

    y_true = dataset.y.astype(y_pred.dtype)

    labels = np.unique(y_true)
    labels = sorted(labels.tolist())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    text = None
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        text = [
            [
                f"<b>True Negatives (TN)</b><br />{tn}",
                f"<b>False Positives (FP)</b><br />{fp}",
            ],
            [
                f"<b>False Negatives (FN)</b><br />{fn}",
                f"<b>True Positives (TP)</b><br />{tp}",
            ],
        ]

    fig = ff.create_annotated_heatmap(
        z=cm,
        colorscale="Blues",
        x=labels,
        y=labels,
        annotation_text=text,
    )

    fig["data"][0][
        "hovertemplate"
    ] = "True Label:%{y}<br>Predicted Label:%{x}<br>Count:%{z}<extra></extra>"

    fig.update_layout(
        xaxis=dict(title="Predicted label"),
        yaxis=dict(title="True label"),
        autosize=False,
        width=600,
        height=600,
        title_text="Confusion Matrix",
    )

    fig.add_annotation(
        x=0.5,
        y=-0.1,
        xref="paper",
        yref="paper",
        text=f"Confusion Matrix for {model.input_id} on {dataset.input_id}",
        showarrow=False,
        font=dict(size=14),
    )

    return fig, RawData(
        confusion_matrix=cm,
        threshold=threshold,
        dataset=dataset.input_id,
        model=model.input_id,
    )
