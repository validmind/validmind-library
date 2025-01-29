# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity


def create_stability_analysis_result(
    original_embeddings,
    perturbed_embeddings,
    mean_similarity_threshold=0.7,
):
    # Compute cosine similarities between original and perturbed embeddings
    similarities = cosine_similarity(
        original_embeddings, perturbed_embeddings
    ).diagonal()

    # Store raw data in a dictionary
    raw_data = {
        "original_embeddings": original_embeddings,
        "perturbed_embeddings": perturbed_embeddings,
        "similarities": similarities,
    }

    mean = np.mean(similarities)
    passed = mean > mean_similarity_threshold

    return raw_data, (
        [
            {
                "Mean Similarity": mean,
                "Min Similarity": np.min(similarities),
                "Max Similarity": np.max(similarities),
                "Median Similarity": np.median(similarities),
                "Std Similarity": np.std(similarities),
                "Pass/Fail": "Pass" if passed else "Fail",
            }
        ],
        px.histogram(
            x=similarities.flatten(),
            nbins=100,
            title="Cosine Similarity Distribution",
            labels={"x": "Cosine Similarity"},
        ),
        px.density_contour(
            x=similarities.flatten(),
            nbinsx=100,
            title="Cosine Similarity Density",
            labels={"x": "Cosine Similarity"},
            marginal_x="histogram",
        ),
        px.box(
            x=similarities.flatten(),
            labels={"x": "Cosine Similarity"},
            title="Cosine Similarity Box Plot",
        ),
        passed,
    )
