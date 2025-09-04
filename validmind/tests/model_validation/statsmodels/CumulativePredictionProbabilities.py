# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from matplotlib import cm

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk")
@tasks("classification")
def CumulativePredictionProbabilities(
    dataset: VMDataset, model: VMModel, title: str = "Cumulative Probabilities"
) -> Tuple[go.Figure, RawData]:
    """
    Visualizes cumulative probabilities of positive and negative classes for both training and testing in classification models.

    ### Purpose

    This metric is utilized to evaluate the distribution of predicted probabilities for positive and negative classes
    in a classification model. It provides a visual assessment of the model's behavior by plotting the cumulative
    probabilities for positive and negative classes across both the training and test datasets.

    ### Test Mechanism

    The classification model is evaluated by first computing the predicted probabilities for each instance in both
    the training and test datasets, which are then added as a new column in these sets. The cumulative probabilities
    for positive and negative classes are subsequently calculated and sorted in ascending order. Cumulative
    distributions of these probabilities are created for both positive and negative classes across both training and
    test datasets. These cumulative probabilities are represented visually in a plot, containing two subplots - one for
    the training data and the other for the test data, with lines representing cumulative distributions of positive and
    negative classes.

    ### Signs of High Risk

    - Imbalanced distribution of probabilities for either positive or negative classes.
    - Notable discrepancies or significant differences between the cumulative probability distributions for the
    training data versus the test data.
    - Marked discrepancies or large differences between the cumulative probability distributions for positive and
    negative classes.

    ### Strengths

    - Provides a visual illustration of data, which enhances the ease of understanding and interpreting the model's
    behavior.
    - Allows for the comparison of model's behavior across training and testing datasets, providing insights about how
    well the model is generalized.
    - Differentiates between positive and negative classes and their respective distribution patterns, aiding in
    problem diagnosis.

    ### Limitations

    - Exclusive to classification tasks and specifically to classification models.
    - Graphical results necessitate human interpretation and may not be directly applicable for automated risk
    detection.
    - The method does not give a solitary quantifiable measure of model risk, instead, it offers a visual
    representation and broad distributional information.
    - If the training and test datasets are not representative of the overall data distribution, the metric could
    provide misleading results.
    """

    df = dataset.df
    df["probabilities"] = dataset.y_prob(model)

    fig, fig_data = _plot_cumulative_prob(df, dataset.target_column, title)

    return fig, RawData(
        cumulative_probabilities=fig_data,
        model=model.input_id,
        dataset=dataset.input_id,
    )


def _plot_cumulative_prob(df, target_col, title):

    # Generate a colormap and convert to Plotly-accepted color format
    # Adjust 'viridis' to any other matplotlib colormap if desired
    colormap = cm.get_cmap("viridis")

    fig = go.Figure()

    # Get unique classes and assign colors
    classes = sorted(df[target_col].unique())
    colors = [colormap(i / len(classes))[:3] for i in range(len(classes))]  # RGB
    color_dict = {
        cls: f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
        for cls, rgb in zip(classes, colors)
    }

    raw_data = {}

    for class_value in sorted(df[target_col].unique()):
        # Calculate cumulative distribution for the current class
        sorted_probs = np.sort(df[df[target_col] == class_value]["probabilities"])
        cumulative_probs = np.cumsum(sorted_probs) / np.sum(sorted_probs)
        raw_data[class_value] = {
            "sorted_probs": sorted_probs,
            "cumulative_probs": cumulative_probs,
        }

        fig.add_trace(
            go.Scatter(
                x=sorted_probs,
                y=cumulative_probs,
                mode="lines",
                name=f"{target_col} = {class_value}",
                line=dict(
                    color=color_dict[class_value],
                ),
            )
        )
        fig.update_layout(
            title_text=f"{title}",
            xaxis_title="Probability",
            yaxis_title="Cumulative Distribution",
        )

    return fig, raw_data
