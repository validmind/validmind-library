# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def MinimumROCAUCScore(
    dataset: VMDataset, model: VMModel, min_threshold: float = 0.5
) -> Tuple[List[Dict[str, float]], bool, RawData]:
    """
    Validates model by checking if the ROC AUC score meets or surpasses a specified threshold.

    ### Purpose

    The Minimum ROC AUC Score test is used to determine the model's performance by ensuring that the Receiver Operating
    Characteristic Area Under the Curve (ROC AUC) score on the validation dataset meets or exceeds a predefined
    threshold. The ROC AUC score indicates how well the model can distinguish between different classes, making it a
    crucial measure in binary and multiclass classification tasks.

    ### Test Mechanism

    This test implementation calculates the multiclass ROC AUC score on the true target values and the model's
    predictions. The test converts the multi-class target variables into binary format using `LabelBinarizer` before
    computing the score. If this ROC AUC score is higher than the predefined threshold (defaulted to 0.5), the test
    passes; otherwise, it fails. The results, including the ROC AUC score, the threshold, and whether the test passed
    or failed, are then stored in a `ThresholdTestResult` object.

    ### Signs of High Risk

    - A high risk or failure in the model's performance as related to this metric would be represented by a low ROC AUC
    score, specifically any score lower than the predefined minimum threshold. This suggests that the model is
    struggling to distinguish between different classes effectively.

    ### Strengths

    - The test considers both the true positive rate and false positive rate, providing a comprehensive performance
    measure.
    - ROC AUC score is threshold-independent meaning it measures the model's quality across various classification
    thresholds.
    - Works robustly with binary as well as multi-class classification problems.

    ### Limitations

    - ROC AUC may not be useful if the class distribution is highly imbalanced; it could perform well in terms of AUC
    but still fail to predict the minority class.
    - The test does not provide insight into what specific aspects of the model are causing poor performance if the ROC
    AUC score is unsatisfactory.
    - The use of macro average for multiclass ROC AUC score implies equal weightage to each class, which might not be
    appropriate if the classes are imbalanced.
    """
    y_true = dataset.y

    if len(np.unique(y_true)) > 2:
        lb = LabelBinarizer()
        lb.fit(y_true)

        y_true_binary = lb.transform(y_true)
        y_score_binary = lb.transform(dataset.y_pred(model))

        roc_auc = roc_auc_score(
            y_true=y_true_binary,
            y_score=y_score_binary,
            average="macro",
        )

    else:
        y_score_prob = dataset.y_prob(model)
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_score_prob)

    results = [
        {
            "Score": roc_auc,
            "Threshold": min_threshold,
            "Pass/Fail": "Pass" if roc_auc > min_threshold else "Fail",
        }
    ]

    return (
        results,
        roc_auc > min_threshold,
        RawData(
            y_true=y_true,
            roc_auc=roc_auc,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
