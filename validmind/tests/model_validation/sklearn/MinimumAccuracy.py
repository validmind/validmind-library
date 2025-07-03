# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score

from validmind import RawData
from validmind.tests import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags(
    "sklearn", "binary_classification", "multiclass_classification", "model_performance"
)
@tasks("classification", "text_classification")
def MinimumAccuracy(
    dataset: VMDataset, model: VMModel, min_threshold: float = 0.7
) -> Tuple[List[Dict[str, float]], bool, RawData]:
    """
    Checks if the model's prediction accuracy meets or surpasses a specified threshold.

    ### Purpose

    The Minimum Accuracy test’s objective is to verify whether the model's prediction accuracy on a specific dataset
    meets or surpasses a predetermined minimum threshold. Accuracy, which is simply the ratio of correct predictions to
    total predictions, is a key metric for evaluating the model's performance. Considering binary as well as multiclass
    classifications, accurate labeling becomes indispensable.

    ### Test Mechanism

    The test mechanism involves contrasting the model's accuracy score with a preset minimum threshold value, with the
    default being 0.7. The accuracy score is computed utilizing sklearn’s `accuracy_score` method, where the true
    labels `y_true` and predicted labels `class_pred` are compared. If the accuracy score is above the threshold, the
    test receives a passing mark. The test returns the result along with the accuracy score and threshold used for the
    test.

    ### Signs of High Risk

    - Model fails to achieve or surpass the predefined score threshold.
    - Persistent scores below the threshold, indicating a high risk of inaccurate predictions.

    ### Strengths

    - Simplicity, presenting a straightforward measure of holistic model performance across all classes.
    - Particularly advantageous when classes are balanced.
    - Versatile, as it can be implemented on both binary and multiclass classification tasks.

    ### Limitations

    - Misleading accuracy scores when classes in the dataset are highly imbalanced.
    - Favoritism towards the majority class, giving an inaccurate perception of model performance.
    - Inability to measure the model's precision, recall, or capacity to manage false positives or false negatives.
    - Focused on overall correctness and may not be sufficient for all types of model analytics.
    """
    accuracy = accuracy_score(dataset.y, dataset.y_pred(model))

    return (
        [
            {
                "Score": accuracy,
                "Threshold": min_threshold,
                "Pass/Fail": "Pass" if accuracy > min_threshold else "Fail",
            }
        ],
        accuracy > min_threshold,
        RawData(model=model.input_id, dataset=dataset.input_id),
    )
