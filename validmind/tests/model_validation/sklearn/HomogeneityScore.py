# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from sklearn import metrics

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance")
@tasks("clustering")
def HomogeneityScore(
    dataset: VMDataset, model: VMModel
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Assesses clustering homogeneity by comparing true and predicted labels, scoring from 0 (heterogeneous) to 1
    (homogeneous).

    ### Purpose

    The Homogeneity Score encapsulated in this performance test is used to measure the homogeneity of the clusters
    formed by a machine learning model. In simple terms, a clustering result satisfies homogeneity if all of its
    clusters contain only points which are members of a single class.

    ### Test Mechanism

    This test uses the `homogeneity_score` function from the `sklearn.metrics` library to compare the ground truth
    class labels of the training and testing sets with the labels predicted by the given model. The returned score is a
    metric of the clustering accuracy, and ranges from 0.0 to 1.0, with 1.0 denoting the highest possible degree of
    homogeneity.

    ### Signs of High Risk

    - A score close to 0: This denotes that clusters are highly heterogenous and points within the same cluster might
    not belong to the same class.
    - A significantly lower score for testing data compared to the score for training data: This can indicate
    overfitting, where the model has learned to perfectly match the training data but fails to perform well on unseen
    data.

    ### Strengths

    - It provides a simple quantitative measure of the degree to which clusters contain points from only one class.
    - Useful for validating clustering solutions where the ground truth — class membership of points — is known.
    - It's agnostic to the absolute labels, and cares only that the points within the same cluster have the same class
    label.

    ### Limitations

    - The Homogeneity Score is not useful for clustering solutions where the ground truth labels are not known.
    - It doesn’t work well with differently sized clusters since it gives predominance to larger clusters.
    - The score does not address the actual number of clusters formed, or the evenness of cluster sizes. It only checks
    the homogeneity within the given clusters created by the model.
    """
    homogeneity_score = metrics.homogeneity_score(
        labels_true=dataset.y,
        labels_pred=dataset.y_pred(model),
    )

    raw_data = RawData(
        y_true=dataset.y,
        y_pred=dataset.y_pred(model),
        model=model.input_id,
        dataset=dataset.input_id,
    )

    return ([{"Homogeneity Score": homogeneity_score}], raw_data)
