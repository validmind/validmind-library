# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "dimensionality_reduction", "embeddings")
@tasks("text_qa", "text_generation", "text_summarization")
def EuclideanDistanceComparison(
    dataset: VMDataset, models: List[VMModel]
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Assesses and visualizes the dissimilarity between model embeddings using Euclidean distance, providing insights
    into model behavior and potential redundancy or diversity.

    ### Purpose

    The Euclidean Distance Comparison test aims to analyze and compare the embeddings produced by different models. By
    measuring the Euclidean distance between vectors in Euclidean space, it provides a metric to assess the magnitude
    of dissimilarity between embeddings created by different models. This is crucial for tasks that require models to
    produce distinct responses or feature separations.

    ### Test Mechanism

    The test computes the embeddings for each model using the provided dataset and calculates the Euclidean distance
    for every possible pair of models. It generates a distance matrix where each element represents the Euclidean
    distance between two model embeddings. This matrix is then visualized through bar charts, showing the distance
    distribution for each model pair. Additionally, it compiles a table with descriptive statistics such as mean,
    median, standard deviation, minimum, and maximum distances for each model pair, including references to the
    compared models.

    ### Signs of High Risk

    - Very high distance values could suggest that models are focusing on entirely different features or aspects of the
    data, which might be undesirable for ensemble methods or when a consensus is required.
    - Extremely low distances across different models might indicate redundancy, suggesting that models are not
    providing diverse enough perspectives on the data.

    ### Strengths

    - Provides a clear and quantifiable measure of how different the embeddings from various models are.
    - Useful for identifying outlier models or those that behave significantly differently from others in a group.

    ### Limitations

    - Euclidean distance can be sensitive to the scale of the data, meaning that preprocessing steps like normalization
    might be necessary to ensure meaningful comparisons.
    - Does not consider the orientation or angle between vectors, focusing purely on magnitude differences.
    """

    figures = []
    all_stats = []

    distance_matrices = []

    # Generate all pairs of models for comparison
    for model_A, model_B in combinations(models, 2):
        embeddings_A = np.stack(dataset.y_pred(model_A))
        embeddings_B = np.stack(dataset.y_pred(model_B))

        # Calculate pairwise Euclidean distances
        distance_matrix = euclidean_distances(embeddings_A, embeddings_B)
        distances = distance_matrix.flatten()

        # Store raw distance matrix for each pair-wise comparison
        distance_matrices.append(
            {
                "model_A": model_A.input_id,
                "model_B": model_B.input_id,
                "distance_matrix": distance_matrix,
            }
        )

        # Generate statistics and add model combination as a column
        stats_data = {
            "Combination": f"{model_A.input_id} vs {model_B.input_id}",
            "Mean": np.mean(distances),
            "Median": np.median(distances),
            "Standard Deviation": np.std(distances),
            "Minimum": np.min(distances),
            "Maximum": np.max(distances),
        }
        all_stats.append(stats_data)

        # Generate an index for each distance value
        indices = range(len(distances))

        # Create the bar chart using Plotly
        fig = px.bar(
            x=indices,
            y=distances,
            labels={"x": "Pair Index", "y": "Euclidean Distance"},
            title=f"Euclidean Distance - {model_A.input_id} vs {model_B.input_id}",
        )
        fig.update_layout(xaxis_title="Pair Index", yaxis_title="Euclidean Distance")
        figures.append(fig)

    # Create a DataFrame from all collected statistics
    stats_df = pd.DataFrame(all_stats)

    # Add raw data to return
    raw_data = RawData(
        distance_matrices=pd.DataFrame(distance_matrices),
        dataset=dataset.input_id,
        models=[model.input_id for model in models],
    )

    return (stats_df, *figures, raw_data)
