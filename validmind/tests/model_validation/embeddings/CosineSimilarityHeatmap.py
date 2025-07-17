# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "dimensionality_reduction", "embeddings")
@tasks("text_qa", "text_generation", "text_summarization")
def CosineSimilarityHeatmap(
    dataset: VMDataset,
    model: VMModel,
    title="Cosine Similarity Matrix",
    color="Cosine Similarity",
    xaxis_title="Index",
    yaxis_title="Index",
    color_scale="Blues",
) -> Tuple[go.Figure, RawData]:
    """
    Generates an interactive heatmap to visualize the cosine similarities among embeddings derived from a given model.

    ### Purpose

    This function is designed to visually analyze the cosine similarities of embeddings from a specific model. Cosine
    similarity, a measure of the cosine of the angle between two vectors, aids in understanding the orientation and
    similarity of vectors in multi-dimensional space. This is particularly valuable for exploring text embeddings and
    their relative similarities among documents, words, or phrases.

    ### Test Mechanism

    The function operates through a sequence of steps to visualize cosine similarities. Initially, embeddings are
    extracted for each dataset entry using the designated model. Following this, the function computes the pairwise
    cosine similarities among these embeddings. The computed similarities are then displayed in an interactive heatmap.

    ### Signs of High Risk

    - High similarity values (close to 1) across the heatmap might not always be indicative of a risk; however, in
    contexts where diverse perspectives or features are desired, this could suggest a lack of diversity in the model's
    learning process or potential redundancy.
    - Similarly, low similarity values (close to -1) indicate strong dissimilarity, which could be beneficial in
    scenarios demanding diverse outputs. However, in cases where consistency is needed, these low values might
    highlight that the model is unable to capture a coherent set of features from the data, potentially leading to poor
    performance on related tasks.

    ### Strengths

    - Provides an interactive and intuitive visual representation of embedding similarities, facilitating easy
    exploration and analysis.
    - Allows customization of visual elements such as title, axis labels, and color scale to suit specific analytical
    needs and preferences.

    ### Limitations

    - As the number of embeddings increases, the effectiveness of the heatmap might diminish due to overcrowding,
    making it hard to discern detailed similarities.
    - The interpretation of the heatmap heavily relies on the appropriate setting of the color scale, as incorrect
    settings can lead to misleading visual interpretations.
    """

    embeddings = np.stack(dataset.y_pred(model))

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Create the heatmap using Plotly
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x=xaxis_title, y=yaxis_title, color=color),
        text_auto=True,
        aspect="auto",
        color_continuous_scale=color_scale,
    )

    fig.update_layout(
        title=f"{title} - {model.input_id}",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return fig, RawData(
        similarity_matrix=similarity_matrix,
        model=model.input_id,
        dataset=dataset.input_id,
    )
