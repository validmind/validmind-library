# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import silhouette_samples, silhouette_score

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance")
@tasks("clustering")
def SilhouettePlot(
    model: VMModel, dataset: VMDataset
) -> Tuple[Dict[str, float], go.Figure, RawData]:
    """
    Calculates and visualizes Silhouette Score, assessing the degree of data point suitability to its cluster in ML
    models.

    ### Purpose

    This test calculates the Silhouette Score, which is a model performance metric used in clustering applications.
    Primarily, the Silhouette Score evaluates how similar a data point is to its own cluster compared to other
    clusters. The metric ranges between -1 and 1, where a high value indicates that the object is well matched to its
    own cluster and poorly matched to neighboring clusters. Thus, the goal is to achieve a high Silhouette Score,
    implying well-separated clusters.

    ### Test Mechanism

    The test first extracts the true and predicted labels from the model's training data. The test runs the Silhouette
    Score function, which takes as input the training dataset features and the predicted labels, subsequently
    calculating the average score. This average Silhouette Score is printed for reference. The script then calculates
    the silhouette coefficients for each data point, helping to form the Silhouette Plot. Each cluster is represented
    in this plot, with color distinguishing between different clusters. A red dashed line indicates the average
    Silhouette Score. The Silhouette Scores are also collected into a structured table, facilitating model performance
    analysis and comparison.

    ### Signs of High Risk

    - A low Silhouette Score, potentially indicating that the clusters are not well separated and that data points may
    not be fitting well to their respective clusters.
    - A Silhouette Plot displaying overlapping clusters or the absence of clear distinctions between clusters visually
    also suggests poor clustering performance.

    ### Strengths

    - The Silhouette Score provides a clear and quantitative measure of how well data points have been grouped into
    clusters, offering insights into model performance.
    - The Silhouette Plot provides an intuitive, graphical representation of the clustering mechanism, aiding visual
    assessments of model performance.
    - It does not require ground truth labels, so it's useful when true cluster assignments are not known.

    ### Limitations

    - The Silhouette Score may be susceptible to the influence of outliers, which could impact its accuracy and
    reliability.
    - It assumes the clusters are convex and isotropic, which might not be the case with complex datasets.
    - Due to the average nature of the Silhouette Score, the metric does not account for individual data point
    assignment nuances, so potentially relevant details may be omitted.
    - Computationally expensive for large datasets, as it requires pairwise distance computations.
    """
    y_pred = dataset.y_pred(model)

    silhouette_avg = silhouette_score(
        X=dataset.x,
        labels=y_pred,
        metric="euclidean",
    )

    # Calculate silhouette coefficients for each data point
    sample_silhouette_values = silhouette_samples(dataset.x, y_pred)
    # Create a silhouette plot
    fig, ax = plt.subplots()

    y_lower = 10
    num_clusters = len(np.unique(y_pred))
    for i in range(num_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[y_pred == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.viridis(float(i) / num_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for the next plot
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot for Clusters")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")

    # The vertical line represents the average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.close()

    return (
        {
            "Silhouette Score": silhouette_avg,
        },
        fig,
        RawData(
            sample_silhouette_values=sample_silhouette_values,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
