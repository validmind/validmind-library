# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

from validmind import RawData, tags, tasks
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

logger = get_logger(__name__)


@tags("statsmodels", "feature_importance", "visualization")
@tasks("regression")
def RegressionPermutationFeatureImportance(
    dataset: VMDataset, model: VMModel, fontsize: int = 12, figure_height: int = 500
) -> Tuple[go.Figure, RawData]:
    """
    Assesses the significance of each feature in a model by evaluating the impact on model performance when feature
    values are randomly rearranged.

    ### Purpose

    The primary purpose of this metric is to determine which features significantly impact the performance of a
    regression model developed using statsmodels. The metric measures how much the prediction accuracy deteriorates
    when each feature's values are permuted.

    ### Test Mechanism

    This metric shuffles the values of each feature one at a time in the dataset, computes the model's performance
    after each permutation, and compares it to the baseline performance. A significant decrease in performance
    indicates the importance of the feature.

    ### Signs of High Risk

    - Significant reliance on a feature that, when permuted, leads to a substantial decrease in performance, suggesting
    overfitting or high model dependency on that feature.
    - Features identified as unimportant despite known impacts from domain knowledge, suggesting potential issues in
    model training or data preprocessing.

    ### Strengths

    - Directly assesses the impact of each feature on model performance, providing clear insights into model
    dependencies.
    - Model-agnostic within the scope of statsmodels, applicable to any regression model that outputs predictions.

    ### Limitations

    - The metric is specific to statsmodels and cannot be used with other types of models without adaptation.
    - It does not capture interactions between features, which can lead to underestimating the importance of correlated
    features.
    - Assumes independence of features when calculating importance, which might not always hold true.
    """
    y_true = dataset.y

    baseline_performance = r2_score(y_true, dataset.y_pred(model))

    importances = pd.DataFrame(
        index=dataset.feature_columns, columns=["Importance", "Std Dev"]
    )

    for column in dataset.feature_columns:
        shuffled_scores = []
        for _ in range(30):  # Default number of shuffles
            x_shuffled = dataset.x_df()
            x_shuffled[column] = check_random_state(0).permutation(x_shuffled[column])
            permuted_performance = r2_score(y_true, model.predict(x_shuffled))
            shuffled_scores.append(baseline_performance - permuted_performance)

        importances.loc[column] = {
            "Importance": np.mean(shuffled_scores),
            "Std Dev": np.std(shuffled_scores),
        }

    sorted_idx = importances["Importance"].argsort()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=importances.index[sorted_idx],
            x=importances.loc[importances.index[sorted_idx], "Importance"],
            orientation="h",
            error_x=dict(
                type="data",
                array=importances.loc[importances.index[sorted_idx], "Std Dev"],
            ),
        )
    )
    fig.update_layout(
        title_text="Permutation Feature Importances",
        yaxis=dict(tickmode="linear", dtick=1, tickfont=dict(size=fontsize)),
        height=figure_height,
    )

    return fig, RawData(
        importances=importances, model=model.input_id, dataset=dataset.input_id
    )
