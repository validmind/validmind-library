# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score

from validmind import RawData
from validmind.tests import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def MinimumF1Score(
    dataset: VMDataset, model: VMModel, min_threshold: float = 0.5
) -> Tuple[List[Dict[str, float]], bool, RawData]:
    """
    Assesses if the model's F1 score on the validation set meets a predefined minimum threshold, ensuring balanced
    performance between precision and recall.

    ### Purpose

    The main objective of this test is to ensure that the F1 score, a balanced measure of precision and recall, of the
    model meets or surpasses a predefined threshold on the validation dataset. The F1 score is highly useful for
    gauging model performance in classification tasks, especially in cases where the distribution of positive and
    negative classes is skewed.

    ### Test Mechanism

    The F1 score for the validation dataset is computed through scikit-learn's metrics in Python. The scoring mechanism
    differs based on the classification problem: for multi-class problems, macro averaging is used, and for binary
    classification, the built-in `f1_score` calculation is used. The obtained F1 score is then assessed against the
    predefined minimum F1 score that is expected from the model.

    ### Signs of High Risk

    - If a model returns an F1 score that is less than the established threshold, it is regarded as high risk.
    - A low F1 score might suggest that the model is not finding an optimal balance between precision and recall,
    failing to effectively identify positive classes while minimizing false positives.

    ### Strengths

    - Provides a balanced measure of a model's performance by accounting for both false positives and false negatives.
    - Particularly advantageous in scenarios with imbalanced class distribution, where accuracy can be misleading.
    - Flexibility in setting the threshold value allows tailored minimum acceptable performance standards.

    ### Limitations

    - May not be suitable for all types of models and machine learning tasks.
    - The F1 score assumes an equal cost for false positives and false negatives, which may not be true in some
    real-world scenarios.
    - Practitioners might need to rely on other metrics such as precision, recall, or the ROC-AUC score that align more
    closely with specific requirements.
    """

    if len(np.unique(dataset.y)) > 2:
        score = f1_score(dataset.y, dataset.y_pred(model), average="macro")
    else:
        score = f1_score(dataset.y, dataset.y_pred(model))

    return (
        [
            {
                "Score": score,
                "Threshold": min_threshold,
                "Pass/Fail": "Pass" if score > min_threshold else "Fail",
            }
        ],
        score > min_threshold,
        RawData(score=score, model=model.input_id, dataset=dataset.input_id),
    )
