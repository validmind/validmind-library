# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from sklearn import metrics

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance")
@tasks("clustering")
def VMeasure(
    dataset: VMDataset, model: VMModel
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Evaluates homogeneity and completeness of a clustering model using the V Measure Score.

    ### Purpose

    The purpose of this metric, V Measure Score (V Score), is to evaluate the performance of a clustering model. It
    measures the homogeneity and completeness of a set of cluster labels, where homogeneity refers to each cluster
    containing only members of a single class and completeness meaning all members of a given class are assigned to the
    same cluster.

    ### Test Mechanism

    ClusterVMeasure is a class that inherits from another class, ClusterPerformance. It uses the `v_measure_score`
    function from the sklearn module's metrics package. The required inputs to perform this metric are the model, train
    dataset, and test dataset. The test is appropriate for models tasked with clustering.

    ### Signs of High Risk

    - Low V Measure Score: A low V Measure Score indicates that the clustering model has poor homogeneity or
    completeness, or both. This might signal that the model is failing to correctly cluster the data.

    ### Strengths

    - The V Measure Score is a harmonic mean between homogeneity and completeness. This ensures that both attributes
    are taken into account when evaluating the model, providing an overall measure of its cluster validity.
    - The metric does not require knowledge of the ground truth classes when measuring homogeneity and completeness,
    making it applicable in instances where such information is unavailable.

    ### Limitations

    - The V Measure Score can be influenced by the number of clusters, which means that it might not always reflect the
    quality of the clustering. Partitioning the data into many small clusters could lead to high homogeneity but low
    completeness, leading to a low V Measure Score even if the clustering might be useful.
    - It assumes equal importance of homogeneity and completeness. In some applications, one may be more important than
    the other. The V Measure Score does not provide flexibility in assigning different weights to homogeneity and
    completeness.
    """
    v_measure = metrics.v_measure_score(
        labels_true=dataset.y,
        labels_pred=dataset.y_pred(model),
    )

    return (
        [{"V Measure": v_measure}],
        RawData(
            v_measure_score=v_measure, model=model.input_id, dataset=dataset.input_id
        ),
    )
