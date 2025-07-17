# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("llm", "text_data", "embeddings", "visualization")
@tasks("feature_extraction")
def ClusterDistribution(
    model: VMModel, dataset: VMDataset, num_clusters: int = 5
) -> Tuple[go.Figure, RawData]:
    """
    Assesses the distribution of text embeddings across clusters produced by a model using KMeans clustering.

    ### Purpose

    The purpose of this metric is to analyze the distribution of the clusters produced by a text embedding model. By
    dividing the text embeddings into different clusters, we can understand how the model is grouping or categorizing
    the text data. This aids in visualizing the organization and segregation of the data, thereby giving an
    understanding of how the model is processing the data.

    ### Test Mechanism

    The metric applies the KMeans clustering algorithm on the predictions made by the model on the testing dataset and
    divides the text embeddings into a pre-defined number of clusters. By default, this number is set to 5 but can be
    customized as per requirements. The output of this test is a histogram plot that shows the distribution of
    embeddings across these clusters.

    ### Signs of High Risk

    - If the embeddings are skewed towards one or two clusters, it indicates that the model is not effectively
    differentiating the various categories in the text data.
    - Uniform distribution of the embeddings across the clusters might show a lack of proper categorization.

    ### Strengths

    - Great tool to visualize the text data categorization by the model.
    - Provides a way to assess if the model is distinguishing the categories effectively or not.
    - Flexible with the number of clusters, so it can be used on various types of data regardless of the number of
    categories.

    ### Limitations

    - Success or failure of this test is based on visual interpretation, which might not be enough for making solid
    conclusions or determining the exact points of failure.
    - Assumes that the division of text embeddings across clusters should ideally be homogeneous, which might not
    always be the case depending on the nature of the text data.
    - Only applies to text embedding models, reducing its utility across various ML models.
    - Uses the KMeans clustering algorithm, which assumes that clusters are convex and isotropic, and may not work as
    intended if the true clusters in the data are not of this shape.
    """
    embeddings = dataset.y_pred(model)
    kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
    labels = kmeans.labels_

    fig = px.histogram(
        labels,
        nbins=num_clusters,
        title="Embeddings Cluster Distribution",
    )

    return fig, RawData(labels=labels, model=model.input_id, dataset=dataset.input_id)
