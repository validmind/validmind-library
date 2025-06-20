# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import List, Tuple

import matplotlib.pyplot as plt

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization")
@tasks("monitoring")
def PredictionAcrossEachFeature(
    datasets: List[VMDataset], model: VMModel
) -> Tuple[plt.Figure, RawData]:
    """
    Assesses differences in model predictions across individual features between reference and monitoring datasets
    through visual analysis.

    ### Purpose

    The Prediction Across Each Feature test aims to visually compare model predictions for each feature between
    reference (training) and monitoring (production) datasets. It helps identify significant differences in prediction
    patterns for further investigation and ensures the model's consistency and stability over time.

    ### Test Mechanism

    The test generates scatter plots for each feature, comparing prediction probabilities between the reference and
    monitoring datasets. Each plot consists of two subplots: one for reference data and one for monitoring data,
    enabling visual comparison of the model's predictive behavior.

    ### Signs of High Risk

    - Significant discrepancies between the reference and monitoring subplots for the same feature.
    - Unexpected patterns or trends in monitoring data that were absent in reference data.

    ### Strengths

    - Provides a clear visual representation of model performance across different features.
    - Facilitates easy identification of features where the model's predictions have diverged.
    - Enables quick detection of potential model performance issues in production.

    ### Limitations

    - Interpretation of scatter plots can be subjective and may require expertise.
    - Visualizations do not provide quantitative metrics for objective evaluation.
    - May not capture all types of distribution changes or issues with the model's predictions.
    """

    """
    This test shows visually the prediction using reference data and monitoring data
    across each individual feature. If there are significant differences in predictions
    across feature values from reference to monitoring dataset then futher investigation
    is needed as the model is producing predictions that are different then what was
    observed during the training of the model.
    """

    y_prob_reference = datasets[0].y_prob(model)
    y_prob_monitoring = datasets[1].y_prob(model)

    figures_to_save = []
    for column in datasets[0].feature_columns:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey="row")

        ax1, ax2 = axs

        ax1.scatter(datasets[0].df[column], y_prob_reference)
        ax2.scatter(datasets[1].df[column], y_prob_monitoring)

        ax1.set_title("Reference")
        ax1.set_xlabel(column)
        ax1.set_ylabel("Prediction Value")

        ax2.set_title("Monitoring")
        ax2.set_xlabel(column)
        figures_to_save.append(fig)
        plt.close()

    return tuple(figures_to_save), RawData(
        y_prob_reference=y_prob_reference,
        y_prob_monitoring=y_prob_monitoring,
        model=model.input_id,
        dataset_reference=datasets[0].input_id,
        dataset_monitoring=datasets[1].input_id,
    )
