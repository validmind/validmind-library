# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

from sklearn.metrics import adjusted_rand_score

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance", "clustering")
@tasks("clustering")
def AdjustedRandIndex(
    model: VMModel, dataset: VMDataset
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Measures the similarity between two data clusters using the Adjusted Rand Index (ARI) metric in clustering machine
    learning models.

    ### Purpose

    The Adjusted Rand Index (ARI) metric is intended to measure the similarity between two data clusters. This metric
    is specifically used for clustering machine learning models to quantify how well the model is clustering and
    producing data groups. It involves comparing the model's produced clusters against the actual (true) clusters found
    in the dataset.

    ### Test Mechanism

    The Adjusted Rand Index (ARI) is calculated using the `adjusted_rand_score` method from the `sklearn.metrics`
    module in Python. The test requires inputs including the model itself and the model's training and test datasets.
    The model's computed clusters and the true clusters are compared, and the similarities are measured to compute the
    ARI.

    ### Signs of High Risk

    - If the ARI is close to zero, it signifies that the model's cluster assignments are random and do not match the
    actual dataset clusters, indicating a high risk.
    - An ARI of less than zero indicates that the model's clustering performance is worse than random.

    ### Strengths

    - ARI is normalized and provides a consistent metric between -1 and +1, irrespective of raw cluster sizes or
    dataset size variations.
    - It does not require a ground truth for computation, making it ideal for unsupervised learning model evaluations.
    - It penalizes for false positives and false negatives, providing a robust measure of clustering quality.

    ### Limitations

    - In real-world situations, true clustering is often unknown, which can hinder the practical application of the ARI.
    - The ARI requires all individual data instances to be independent, which may not always hold true.
    - It may be difficult to interpret the implications of an ARI score without context or a benchmark, as it is
    heavily dependent on the characteristics of the dataset used.
    """
    ari = adjusted_rand_score(
        labels_true=dataset.y,
        labels_pred=dataset.y_pred(model),
    )

    return [{"Adjusted Rand Index": ari}], RawData(
        ari_score=ari, model=model.input_id, dataset=dataset.input_id
    )
