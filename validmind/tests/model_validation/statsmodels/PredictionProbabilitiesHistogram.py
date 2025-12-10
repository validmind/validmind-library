# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import plotly.graph_objects as go
from matplotlib import cm

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization", "credit_risk")
@tasks("classification")
def PredictionProbabilitiesHistogram(
    dataset: VMDataset,
    model: VMModel,
    title: str = "Histogram of Predictive Probabilities",
) -> Tuple[go.Figure, RawData]:
    """
    Assesses the predictive probability distribution for binary classification to evaluate model performance and
    potential overfitting or bias.

    ### Purpose

    The Prediction Probabilities Histogram test is designed to generate histograms displaying the Probability of
    Default (PD) predictions for both positive and negative classes in training and testing datasets. This helps in
    evaluating the performance of a classification model.

    ### Test Mechanism

    The metric follows these steps to execute the test:
    - Extracts the target column from both the train and test datasets.
    - Uses the model's predict function to calculate probabilities.
    - Adds these probabilities as a new column to the training and testing dataframes.
    - Generates histograms for each class (0 or 1) within the training and testing datasets.
    - Sets different opacities for the histograms to enhance visualization.
    - Overlays the four histograms (two for training and two for testing) on two different subplot frames.
    - Returns a plotly graph object displaying the visualization.

    ### Signs of High Risk

    - Significant discrepancies between the histograms of training and testing data.
    - Large disparities between the histograms for the positive and negative classes.
    - Potential overfitting or bias indicated by significant issues.
    - Unevenly distributed probabilities suggesting inaccurate model predictions.

    ### Strengths

    - Offers a visual representation of the PD predictions made by the model, aiding in understanding its behavior.
    - Assesses both the training and testing datasets, adding depth to model validation.
    - Highlights disparities between classes, providing insights into class imbalance or data skewness.
    - Effectively visualizes risk spread, which is particularly beneficial for credit risk prediction.

    ### Limitations

    - Specifically tailored for binary classification scenarios and not suited for multi-class classification tasks.
    - Provides a robust visual representation but lacks a quantifiable measure to assess model performance.
    """

    df = dataset.df
    df["probabilities"] = dataset.y_prob(model)

    fig = _plot_prob_histogram(df, dataset.target_column, title)

    return fig, RawData(
        probabilities=df["probabilities"],
        model=model.input_id,
        dataset=dataset.input_id,
    )


def _plot_prob_histogram(df, target_col, title):

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

    # Ensure classes are plotted in the specified order
    for class_value in sorted(df[target_col].unique()):
        fig.add_trace(
            go.Histogram(
                x=df[df[target_col] == class_value]["probabilities"],
                opacity=0.75,
                name=f"{target_col} = {class_value}",
                marker=dict(
                    color=color_dict[class_value],
                ),
            )
        )
    fig.update_layout(
        barmode="overlay",
        title_text=f"{title}",
        xaxis_title="Probability",
        yaxis_title="Frequency",
    )

    return fig
