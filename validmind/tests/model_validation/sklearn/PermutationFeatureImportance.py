# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass

import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from validmind.errors import SkipTestError
from validmind.logging import get_logger
from validmind.vm_models import Figure, Metric

logger = get_logger(__name__)


@dataclass
class PermutationFeatureImportance(Metric):
    """
    Assesses the significance of each feature in a model by evaluating the impact on model performance when feature
    values are randomly rearranged.

    ### Purpose

    The Permutation Feature Importance (PFI) metric aims to assess the importance of each feature used by the Machine
    Learning model. The significance is measured by evaluating the decrease in the model's performance when the
    feature's values are randomly arranged.

    ### Test Mechanism

    PFI is calculated via the `permutation_importance` method from the `sklearn.inspection` module. This method
    shuffles the columns of the feature dataset and measures the impact on the model's performance. A significant
    decrease in performance after permutating a feature's values deems the feature as important. On the other hand, if
    performance remains the same, the feature is likely not important. The output of the PFI metric is a figure
    illustrating the importance of each feature.

    ### Signs of High Risk

    - The model heavily relies on a feature with highly variable or easily permutable values, indicating instability.
    - A feature deemed unimportant by the model but expected to have a significant effect on the outcome based on
    domain knowledge is not influencing the model's predictions.

    ### Strengths

    - Provides insights into the importance of different features and may reveal underlying data structure.
    - Can indicate overfitting if a particular feature or set of features overly impacts the model's predictions.
    - Model-agnostic and can be used with any classifier that provides a measure of prediction accuracy before and
    after feature permutation.

    ### Limitations

    - Does not imply causality; it only presents the amount of information that a feature provides for the prediction
    task.
    - Does not account for interactions between features. If features are correlated, the permutation importance may
    allocate importance to one and not the other.
    - Cannot interact with certain libraries like statsmodels, pytorch, catboost, etc., thus limiting its applicability.
    """

    name = "pfi"
    required_inputs = ["model", "dataset"]
    default_params = {
        "fontsize": None,
        "figure_height": 1000,
    }
    tasks = ["classification", "text_classification"]
    tags = [
        "sklearn",
        "binary_classification",
        "multiclass_classification",
        "feature_importance",
        "visualization",
    ]

    def run(self):
        x = self.inputs.dataset.x_df()
        y = self.inputs.dataset.y_df()

        if self.inputs.model.library in [
            "statsmodels",
            "pytorch",
            "catboost",
            "transformers",
            "R",
        ]:
            raise SkipTestError(f"Skipping PFI for {self.inputs.model.library} models")

        pfi_values = permutation_importance(
            self.inputs.model.model,
            x,
            y,
            random_state=0,
            n_jobs=-2,
        )

        pfi = {}
        for i, column in enumerate(x.columns):
            pfi[column] = [pfi_values["importances_mean"][i]], [
                pfi_values["importances_std"][i]
            ]

        sorted_idx = pfi_values.importances_mean.argsort()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=x.columns[sorted_idx],
                x=pfi_values.importances[sorted_idx].mean(axis=1).T,
                orientation="h",
            )
        )
        fig.update_layout(
            title_text="Permutation Importances",
            yaxis=dict(
                tickmode="linear",  # set tick mode to linear
                dtick=1,  # set interval between ticks
                tickfont=dict(
                    size=self.params["fontsize"]
                ),  # set the tick label font size
            ),
            height=self.params["figure_height"],  # use figure_height parameter here
        )

        return self.cache_results(
            metric_value=pfi,
            figures=[
                Figure(
                    for_object=self,
                    key=f"pfi_{self.inputs.dataset.input_id}_{self.inputs.model.input_id}",
                    figure=fig,
                ),
            ],
        )
