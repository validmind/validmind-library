# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import itertools
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "dimensionality_reduction", "embeddings")
@tasks("text_qa", "text_generation", "text_summarization")
def PCAComponentsPairwisePlots(
    dataset: VMDataset, model: VMModel, n_components: int = 3
) -> Tuple[go.Figure, RawData]:
    """
    Generates scatter plots for pairwise combinations of principal component analysis (PCA) components of model
    embeddings.

    ### Purpose

    This function visualizes the principal components of embeddings derived from a specified model. Principal Component
    Analysis (PCA) is a statistical technique that emphasizes variation and uncovers strong patterns in a dataset. It
    transforms the original variables into new, uncorrelated variables (principal components) that maximize variance.

    ### Test Mechanism

    The function follows a sequential process to visualize PCA components effectively. It starts by extracting
    embeddings from the dataset, utilizing the model specified by the user. These embeddings are then standardized to
    ensure zero mean and unit variance, which is crucial to prevent any single feature from dominating due to
    scale—this standardization is a critical preprocessing step for PCA. Following this, the function calculates the
    specified number of principal components. The core of the visualization process involves creating scatter plots for
    each pairwise combination of these principal components.

    ### Signs of High Risk

    - If the principal components do not account for a significant portion of the variance, it may suggest that PCA is
    not capturing the essential structures of the data.
    - Similarity in scatter plots across different pairs of components could indicate redundancy in the components,
    suggesting that fewer dimensions might be sufficient to represent the data.

    ### Strengths

    - Enables a simplified visualization of multivariate data, helping to identify patterns across many variables
    effectively.
    - Provides a clear depiction of the directions of maximum variance in the data, which is valuable for feature
    selection and dimensionality reduction.

    ### Limitations

    - PCA's effectiveness hinges on the scaling of the variables; improper standardization can lead to misleading
    interpretations.
    - The interpretation of principal components can be challenging, especially if they capture less significant
    variances or are difficult to relate back to the original features.
    """

    # Get embeddings from the dataset using the model
    embeddings = np.stack(dataset.y_pred(model))

    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(embeddings_scaled)

    # Prepare DataFrame for Plotly
    pca_df = pd.DataFrame(
        pca_results, columns=[f"PC{i + 1}" for i in range(n_components)]
    )

    # List to store each plot
    figures = []

    # Create plots for each pair of principal components
    for pc1, pc2 in itertools.combinations(range(1, n_components + 1), 2):
        fig = px.scatter(
            pca_df,
            x=f"PC{pc1}",
            y=f"PC{pc2}",
            title=f"{getattr(model, 'input_id', 'Unknown Model')} (PC{pc1} vs PC{pc2})",
            labels={
                f"PC{pc1}": f"Principal Component {pc1}",
                f"PC{pc2}": f"Principal Component {pc2}",
            },
        )
        figures.append(fig)

    return (
        *figures,
        RawData(pca_results=pca_df, model=model.input_id, dataset=dataset.input_id),
    )
