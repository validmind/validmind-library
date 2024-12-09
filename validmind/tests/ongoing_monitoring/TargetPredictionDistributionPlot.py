# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import matplotlib.pyplot as plt
import seaborn as sns

from validmind import tags, tasks


@tags("visualization")
@tasks("monitoring")
def TargetPredictionDistributionPlot(datasets, model):
    """
    Assesses differences in prediction distributions between a reference dataset and a monitoring dataset to identify
    potential data drift.

    ### Purpose

    The Target Prediction Distribution Plot test aims to evaluate potential changes in the prediction distributions
    between the reference and new monitoring datasets. It seeks to identify underlying shifts in data characteristics
    that warrant further investigation.

    ### Test Mechanism

    This test generates Kernel Density Estimation (KDE) plots for prediction probabilities from both the reference and
    monitoring datasets. By visually comparing the KDE plots, it assesses significant differences in the prediction
    distributions between the two datasets.

    ### Signs of High Risk

    - Significant divergence between the distribution curves of reference and monitoring predictions.
    - Unusual shifts or bimodal distribution in the monitoring predictions compared to the reference predictions.

    ### Strengths

    - Visual representation makes it easy to spot differences in prediction distributions.
    - Useful for identifying potential data drift or changes in underlying data characteristics.
    - Simple and efficient to implement using standard plotting libraries.

    ### Limitations

    - Subjective interpretation of the visual plots.
    - Might not pinpoint the exact cause of distribution changes.
    - Less effective if the differences in distributions are subtle and not easily visible.
    """

    pred_ref = datasets[0].y_prob_df(model)
    pred_ref.columns = ["Reference Prediction"]
    pred_monitor = datasets[1].y_prob_df(model)
    pred_monitor.columns = ["Monitoring Prediction"]

    fig = plt.figure()
    plot = sns.kdeplot(
        pred_ref["Reference Prediction"], fill=True, label="Reference Prediction"
    )
    plot = sns.kdeplot(
        pred_monitor["Monitoring Prediction"], fill=True, label="Monitor Prediction"
    )
    plot.set(
        xlabel="Prediction", title="Distribution of Reference & Monitor Predictions"
    )
    plot.legend()

    plt.close()

    return fig
