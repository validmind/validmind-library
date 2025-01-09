# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


import matplotlib.pyplot as plt

from validmind import RawData, tags, tasks


@tags("visualization")
@tasks("monitoring")
def PredictionAcrossEachFeature(datasets, model):
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

    df_reference = datasets[0]._df
    df_monitoring = datasets[1]._df

    figures = []
    for column in df_reference:
        prediction_prob_column = f"{model.input_id}_probabilities"
        prediction_column = f"{model.input_id}_prediction"
        if column == prediction_prob_column or column == prediction_column:
            pass
        else:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey="row")

            ax1, ax2 = axs

            ax1.scatter(df_reference[column], df_reference[prediction_prob_column])
            ax2.scatter(df_monitoring[column], df_monitoring[prediction_prob_column])

            ax1.set_title("Reference")
            ax1.set_xlabel(column)
            ax1.set_ylabel("Prediction Value")

            ax2.set_title("Monitoring")
            ax2.set_xlabel(column)
            figures.append(fig)
            plt.close()

    return (*figures, RawData(df_reference=df_reference, df_monitoring=df_monitoring))
