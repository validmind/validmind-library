# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
from sklearn.inspection import permutation_importance

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("model_explainability", "sklearn")
@tasks("regression", "time_series_forecasting")
def FeatureImportance(
    dataset: VMDataset, model: VMModel, num_features: int = 3
) -> Tuple[pd.DataFrame, RawData]:
    """
    Compute feature importance scores for a given model and generate a summary table
    with the top important features.

    ### Purpose

    The Feature Importance Comparison test is designed to compare the feature importance scores for different models
    when applied to various datasets. By doing so, it aims to identify the most impactful features and assess the
    consistency of feature importance across models.

    ### Test Mechanism

    This test works by iterating through each dataset-model pair and calculating permutation feature importance (PFI)
    scores. It then generates a summary table containing the top `num_features` important features for each model. The
    process involves:

    - Extracting features and target data from each dataset.
    - Computing PFI scores using `sklearn.inspection.permutation_importance`.
    - Sorting and selecting the top features based on their importance scores.
    - Compiling these features into a summary table for comparison.

    ### Signs of High Risk

    - Key features expected to be important are ranked low, indicating potential issues with model training or data
    quality.
    - High variance in feature importance scores across different models, suggesting instability in feature selection.

    ### Strengths

    - Provides a clear comparison of the most important features for each model.
    - Uses permutation importance, which is a model-agnostic method and can be applied to any estimator.

    ### Limitations

    - Assumes that the dataset is provided as a DataFrameDataset object with `x_df` and `y_df` methods to access
    feature and target data.
    - Requires that `model.model` is compatible with `sklearn.inspection.permutation_importance`.
    - The function's output is dependent on the number of features specified by `num_features`, which defaults to 3 but
    can be adjusted.
    """
    pfi_values = permutation_importance(
        estimator=model.model,
        X=dataset.x_df(),
        y=dataset.y_df(),
        random_state=0,
        n_jobs=-2,
    )
    pfi = {
        column: pfi_values["importances_mean"][i]
        for i, column in enumerate(dataset.feature_columns)
    }

    # Sort features by their importance
    sorted_features = sorted(pfi.items(), key=lambda item: item[1], reverse=True)
    top_features = sorted_features[:num_features]

    result = {}

    for i in range(num_features):
        if i < len(top_features):
            result[
                f"Feature {i + 1}"
            ] = f"[{top_features[i][0]}; {top_features[i][1]:.4f}]"
        else:
            result[f"Feature {i + 1}"] = None

    return pd.DataFrame([result]), RawData(
        permutation_importance=pfi_values,
        model=model.input_id,
        dataset=dataset.input_id,
    )
