# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


import matplotlib.pyplot as plt
import numpy as np

from validmind import tags, tasks


@tags("visualization")
@tasks("monitoring")
def PredictionCorrelation(datasets, model):
    """
    Assesses correlation changes between model predictions from reference and monitoring datasets to detect potential
    target drift.

    ### Purpose

    To evaluate the changes in correlation pairs between model predictions and features from reference and monitoring
    datasets. This helps in identifying significant shifts that may indicate target drift, potentially affecting model
    performance.

    ### Test Mechanism

    This test calculates the correlation of each feature with model predictions for both reference and monitoring
    datasets. It then compares these correlations side-by-side using a bar plot and a correlation table. Significant
    changes in correlation pairs are highlighted to signal possible model drift.

    ### Signs of High Risk

    - Significant changes in correlation pairs between the reference and monitoring predictions.
    - Notable differences in correlation values, indicating a possible shift in the relationship between features and
    the target variable.

    ### Strengths

    - Provides visual identification of drift in feature relationships with model predictions.
    - Clear bar plot comparison aids in understanding model stability over time.
    - Enables early detection of target drift, facilitating timely interventions.

    ### Limitations

    - Requires substantial reference and monitoring data for accurate comparison.
    - Correlation does not imply causation; other factors may influence changes.
    - Focuses solely on linear relationships, potentially missing non-linear interactions.
    """

    prediction_prob_column = f"{model.input_id}_probabilities"
    prediction_column = f"{model.input_id}_prediction"

    df_corr = datasets[0]._df.corr()
    df_corr = df_corr[[prediction_prob_column]]

    df_corr2 = datasets[1]._df.corr()
    df_corr2 = df_corr2[[prediction_prob_column]]

    corr_final = df_corr.merge(df_corr2, left_index=True, right_index=True)
    corr_final.columns = ["Reference Predictions", "Monitoring Predictions"]
    corr_final = corr_final.drop(index=[prediction_column, prediction_prob_column])

    n = len(corr_final)
    r = np.arange(n)
    width = 0.25

    fig = plt.figure()

    plt.bar(
        r,
        corr_final["Reference Predictions"],
        color="b",
        width=width,
        edgecolor="black",
        label="Reference Prediction Correlation",
    )
    plt.bar(
        r + width,
        corr_final["Monitoring Predictions"],
        color="g",
        width=width,
        edgecolor="black",
        label="Monitoring Prediction Correlation",
    )

    plt.xlabel("Features")
    plt.ylabel("Correlation")
    plt.title("Correlation between Predictions and Features")

    features = corr_final.index.to_list()
    plt.xticks(r + width / 2, features, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.close()

    corr_final["Features"] = corr_final.index
    corr_final = corr_final[
        ["Features", "Reference Predictions", "Monitoring Predictions"]
    ]
    return ({"Correlation Pair Table": corr_final}, fig)
