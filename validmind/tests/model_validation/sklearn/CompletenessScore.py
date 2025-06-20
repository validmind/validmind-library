# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from sklearn.metrics import completeness_score

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance", "clustering")
@tasks("clustering")
def CompletenessScore(
    model: VMModel, dataset: VMDataset
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Evaluates a clustering model's capacity to categorize instances from a single class into the same cluster.

    ### Purpose

    The Completeness Score metric is used to assess the performance of clustering models. It measures the extent to
    which all the data points that are members of a given class are elements of the same cluster. The aim is to
    determine the capability of the model to categorize all instances from a single class into the same cluster.

    ### Test Mechanism

    This test takes three inputs, a model and its associated training and testing datasets. It invokes the
    `completeness_score` function from the sklearn library on the labels predicted by the model. High scores indicate
    that data points from the same class generally appear in the same cluster, while low scores suggest the opposite.

    ### Signs of High Risk

    - Low completeness score: This suggests that the model struggles to group instances from the same class into one
    cluster, indicating poor clustering performance.

    ### Strengths

    - The Completeness Score provides an effective method for assessing the performance of a clustering model,
    specifically its ability to group class instances together.
    - This test metric conveniently relies on the capabilities provided by the sklearn library, ensuring consistent and
    reliable test results.

    ### Limitations

    - This metric only evaluates a specific aspect of clustering, meaning it may not provide a holistic or complete
    view of the model's performance.
    - It cannot assess the effectiveness of the model in differentiating between separate classes, as it is solely
    focused on how well data points from the same class are grouped.
    - The Completeness Score only applies to clustering models; it cannot be used for other types of machine learning
    models.
    """
    score = completeness_score(
        labels_true=dataset.y,
        labels_pred=dataset.y_pred(model),
    )
    return [{"Completeness Score": score}], RawData(
        score=score, model=model.input_id, dataset=dataset.input_id
    )
