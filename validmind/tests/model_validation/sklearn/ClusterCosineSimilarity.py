# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance", "clustering")
@tasks("clustering")
def ClusterCosineSimilarity(
    model: VMModel, dataset: VMDataset
) -> Tuple[List[Dict[str, float]], RawData]:
    """
    Measures the intra-cluster similarity of a clustering model using cosine similarity.

    ### Purpose

    The purpose of this metric is to measure how similar the data points within each cluster of a clustering model are.
    This is done using cosine similarity, which compares the multi-dimensional direction (but not magnitude) of data
    vectors. From a Model Risk Management perspective, this metric is used to quantitatively validate that clusters
    formed by a model have high intra-cluster similarity.

    ### Test Mechanism

    This test works by first extracting the true and predicted clusters of the model's training data. Then, it computes
    the centroid (average data point) of each cluster. Next, it calculates the cosine similarity between each data
    point within a cluster and its respective centroid. Finally, it outputs the mean cosine similarity of each cluster,
    highlighting how similar, on average, data points in a cluster are to the cluster's centroid.

    ### Signs of High Risk

    - Low mean cosine similarity for one or more clusters: If the mean cosine similarity is low, the data points within
    the respective cluster have high variance in their directions. This can be indicative of poor clustering,
    suggesting that the model might not be suitably separating the data into distinct patterns.
    - High disparity between mean cosine similarity values across clusters: If there's a significant difference in mean
    cosine similarity across different clusters, this could indicate imbalance in how the model forms clusters.

    ### Strengths

    - Cosine similarity operates in a multi-dimensional space, making it effective for measuring similarity in high
    dimensional datasets, typical for many machine learning problems.
    - It provides an agnostic view of the cluster performance by only considering the direction (and not the magnitude)
    of each vector.
    - This metric is not dependent on the scale of the variables, making it equally effective on different scales.

    ### Limitations

    - Cosine similarity does not consider magnitudes (i.e. lengths) of vectors, only their direction. This means it may
    overlook instances where clusters have been adequately separated in terms of magnitude.
    - This method summarily assumes that centroids represent the average behavior of data points in each cluster. This
    might not always be true, especially in clusters with high amounts of variance or non-spherical shapes.
    - It primarily works with continuous variables and is not suitable for binary or categorical variables.
    - Lastly, although rare, perfect perpendicular vectors (cosine similarity = 0) could be within the same cluster,
    which may give an inaccurate representation of a 'bad' cluster due to low cosine similarity score.
    """
    y_pred = dataset.y_pred(model)
    num_clusters = len(np.unique(y_pred))

    table = []

    cluster_centroids = {}

    for cluster_idx in range(num_clusters):
        cluster_data = dataset.x[y_pred == cluster_idx]

        if cluster_data.size != 0:
            cluster_centroid = np.mean(cluster_data, axis=0)
            cluster_centroids[cluster_idx] = cluster_centroid
            table.append(
                {
                    "Cluster": cluster_idx,
                    "Mean Cosine Similarity": np.mean(
                        cosine_similarity(
                            X=cluster_data,
                            Y=[cluster_centroid],
                        ).flatten()
                    ),
                }
            )

    if not table:
        raise SkipTestError("No clusters found")

    return table, RawData(
        cluster_centroids=cluster_centroids,
        model=model.input_id,
        dataset=dataset.input_id,
    )
