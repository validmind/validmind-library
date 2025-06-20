# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("llm", "text_data", "embeddings", "visualization")
@tasks("feature_extraction")
def CosineSimilarityDistribution(
    dataset: VMDataset, model: VMModel
) -> Tuple[go.Figure, RawData]:
    """
    Assesses the similarity between predicted text embeddings from a model using a Cosine Similarity distribution
    histogram.

    ### Purpose

    This metric is used to assess the degree of similarity between the embeddings produced by a text embedding model
    using Cosine Similarity. Cosine Similarity is a measure that calculates the cosine of the angle between two
    vectors. This metric is predominantly used in text analysis — in this case, to determine how closely the predicted
    text embeddings align with one another.

    ### Test Mechanism

    The implementation starts by computing the cosine similarity between the predicted values of the model's test
    dataset. These cosine similarity scores are then plotted on a histogram with 100 bins to visualize the distribution
    of the scores. The x-axis of the histogram represents the computed Cosine Similarity.

    ### Signs of High Risk

    - If the cosine similarity scores cluster close to 1 or -1, it may indicate overfitting, as the model's predictions
    are almost perfectly aligned. This could suggest that the model is not generalizable.
    - A broad spread of cosine similarity scores across the histogram may indicate a potential issue with the model's
    ability to generate consistent embeddings.

    ### Strengths

    - Provides a visual representation of the model's performance which is easily interpretable.
    - Can help identify patterns, trends, and outliers in the model's alignment of predicted text embeddings.
    - Useful in measuring the similarity between vectors in multi-dimensional space, important in the case of text
    embeddings.

    ### Limitations

    - Only evaluates the similarity between outputs. It does not provide insight into the model's ability to correctly
    classify or predict.
    - Cosine similarity only considers the angle between vectors and does not consider their magnitude. This can lead
    to high similarity scores for vectors with vastly different magnitudes but a similar direction.
    - The output is sensitive to the choice of bin number for the histogram. Different bin numbers could give a
    slightly altered perspective on the distribution of cosine similarity.
    """
    similarity_scores = cosine_similarity(dataset.y_pred(model)).flatten()

    return px.histogram(
        x=similarity_scores,
        nbins=100,
        title="Cosine Similarity Distribution",
        labels={"x": "Cosine Similarity"},
    ), RawData(
        similarity_scores=similarity_scores,
        model=model.input_id,
        dataset=dataset.input_id,
    )
