# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import clone
from sklearn.metrics import silhouette_score

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError, SkipTestError

try:
    from scipy.spatial.distance import cdist
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for KMeansClustersOptimization. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance", "kmeans")
@tasks("clustering")
def KMeansClustersOptimization(
    model: VMModel, dataset: VMDataset, n_clusters: Optional[List[int]] = None
) -> Tuple[go.Figure, RawData]:
    """
    Optimizes the number of clusters in K-means models using Elbow and Silhouette methods.

    ### Purpose

    This metric is used to optimize the number of clusters used in K-means clustering models. It intends to measure and
    evaluate the optimal number of clusters by leveraging two methodologies, namely the Elbow method and the Silhouette
    method. This is crucial as an inappropriate number of clusters can either overly simplify or overcomplicate the
    structure of the data, thereby undermining the effectiveness of the model.

    ### Test Mechanism

    The test mechanism involves iterating over a predefined range of cluster numbers and applying both the Elbow method
    and the Silhouette method. The Elbow method computes the sum of the minimum euclidean distances between data points
    and their respective cluster centers (distortion). This value decreases as the number of clusters increases; the
    optimal number is typically at the 'elbow' point where the decrease in distortion becomes less pronounced.
    Meanwhile, the Silhouette method calculates the average silhouette score for each data point in the dataset,
    providing a measure of how similar each item is to its own cluster compared to other clusters. The optimal number
    of clusters under this method is the one that maximizes the average silhouette score. The results of both methods
    are plotted for visual inspection.

    ### Signs of High Risk

    - A high distortion value or a low silhouette average score for the optimal number of clusters.
    - No clear 'elbow' point or plateau observed in the distortion plot, or a uniformly low silhouette average score
    across different numbers of clusters, suggesting the data is not amenable to clustering.
    - An optimal cluster number that is unreasonably high or low, suggestive of overfitting or underfitting,
    respectively.

    ### Strengths

    - Provides both a visual and quantitative method to determine the optimal number of clusters.
    - Leverages two different methods (Elbow and Silhouette), thereby affording robustness and versatility in assessing
    the data's clusterability.
    - Facilitates improved model performance by allowing for an informed selection of the number of clusters.

    ### Limitations

    - Assumes that a suitable number of clusters exists in the data, which may not always be true, especially for
    complex or noisy data.
    - Both methods may fail to provide definitive answers when the data lacks clear cluster structures.
    - Might not be straightforward to determine the 'elbow' point or maximize the silhouette average score, especially
    in larger and complicated datasets.
    - Assumes spherical clusters (due to using the Euclidean distance in the Elbow method), which might not align with
    the actual structure of the data.
    """
    if not n_clusters:
        raise SkipTestError(
            "Cluster range must be provided via the 'n_clusters' parameter"
        )

    distortions = {}
    silhouette_avg = {}

    for k in n_clusters:
        kmeanModel = clone(model.model).set_params(n_clusters=k).fit(dataset.x)

        silhouette_avg[k] = silhouette_score(
            dataset.x,
            kmeanModel.predict(dataset.x),
        )

        distortions[k] = (
            sum(
                np.min(
                    cdist(
                        dataset.x,
                        kmeanModel.cluster_centers_,
                        "euclidean",
                    ),
                    axis=1,
                )
            )
            / dataset.x.shape[0]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "The Silhouette value of each cluster",
            "The Elbow Method using Distortion",
        ),
    )

    fig.add_trace(
        go.Scatter(x=list(silhouette_avg.keys()), y=list(silhouette_avg.values())),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Number of clusters", row=1, col=1)
    fig.update_yaxes(title_text="Avg Silhouette Score", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=list(distortions.keys()), y=list(distortions.values())),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Number of clusters", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Distortion", showgrid=False, row=1, col=2)

    fig.update_layout(showlegend=False)

    return fig, RawData(
        distortions=distortions,
        silhouette_avg=silhouette_avg,
        model=model.input_id,
        dataset=dataset.input_id,
    )
