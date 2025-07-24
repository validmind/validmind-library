# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset

logger = get_logger(__name__)


@tags("feature_importance", "AUC", "visualization")
@tasks("classification")
def FeaturesAUC(
    dataset: VMDataset, fontsize: int = 12, figure_height: int = 500
) -> Tuple[go.Figure, RawData]:
    """
    Evaluates the discriminatory power of each individual feature within a binary classification model by calculating
    the Area Under the Curve (AUC) for each feature separately.

    ### Purpose

    The central objective of this metric is to quantify how well each feature on its own can differentiate between the
    two classes in a binary classification problem. It serves as a univariate analysis tool that can help in
    pre-modeling feature selection or post-modeling interpretation.

    ### Test Mechanism

    For each feature, the metric treats the feature values as raw scores to compute the AUC against the actual binary
    outcomes. It provides an AUC value for each feature, offering a simple yet powerful indication of each feature's
    univariate classification strength.

    ### Signs of High Risk

    - A feature with a low AUC score may not be contributing significantly to the differentiation between the two
    classes, which could be a concern if it is expected to be predictive.
    - Conversely, a surprisingly high AUC for a feature not believed to be informative may suggest data leakage or
    other issues with the data.

    ### Strengths

    - By isolating each feature, it highlights the individual contribution of features to the classification task
    without the influence of other variables.
    - Useful for both initial feature evaluation and for providing insights into the model's reliance on individual
    features after model training.

    ### Limitations

    - Does not reflect the combined effects of features or any interaction between them, which can be critical in
    certain models.
    - The AUC values are calculated without considering the model's use of the features, which could lead to different
    interpretations of feature importance when considering the model holistically.
    - This metric is applicable only to binary classification tasks and cannot be directly extended to multiclass
    classification or regression without modifications.
    """
    if len(np.unique(dataset.y)) != 2:
        raise SkipTestError("FeaturesAUC metric requires a binary target variable.")

    aucs = pd.DataFrame(index=dataset.feature_columns, columns=["AUC"])

    for column in dataset.feature_columns:
        feature_values = dataset.df[column]
        if feature_values.nunique() > 1 and pd.api.types.is_numeric_dtype(
            feature_values
        ):
            aucs.loc[column, "AUC"] = roc_auc_score(dataset.y, feature_values)
        else:
            # Not enough unique values to calculate AUC
            aucs.loc[column, "AUC"] = np.nan

    sorted_indices = aucs["AUC"].dropna().sort_values(ascending=False).index

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[column for column in sorted_indices],
            x=[aucs.loc[column, "AUC"] for column in sorted_indices],
            orientation="h",
        )
    )
    fig.update_layout(
        title_text="Feature AUC Scores",
        yaxis=dict(
            tickmode="linear",
            dtick=1,
            tickfont=dict(size=fontsize),
            title="Features",
            autorange="reversed",  # Ensure that the highest AUC is at the top
        ),
        xaxis=dict(title="AUC"),
        height=figure_height,
    )

    return fig, RawData(feature_aucs=aucs, dataset=dataset.input_id)
