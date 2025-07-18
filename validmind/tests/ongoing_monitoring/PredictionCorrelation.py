# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("visualization")
@tasks("monitoring")
def PredictionCorrelation(
    datasets: List[VMDataset],
    model: VMModel,
    drift_pct_threshold: float = 20,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
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

    # Get feature columns and predictions
    feature_columns = datasets[0].feature_columns
    y_prob_ref = pd.Series(datasets[0].y_prob(model), index=datasets[0].df.index)
    y_prob_mon = pd.Series(datasets[1].y_prob(model), index=datasets[1].df.index)

    # Create dataframes with features and predictions
    df_ref = datasets[0].df[feature_columns].copy()
    df_ref["predictions"] = y_prob_ref

    df_mon = datasets[1].df[feature_columns].copy()
    df_mon["predictions"] = y_prob_mon

    # Calculate correlations
    corr_ref = df_ref.corr()["predictions"]
    corr_mon = df_mon.corr()["predictions"]

    # Combine correlations (excluding the predictions row)
    corr_final = pd.DataFrame(
        {
            "Reference Predictions": corr_ref[feature_columns],
            "Monitoring Predictions": corr_mon[feature_columns],
        }
    )

    # Calculate drift percentage with direction
    corr_final["Drift (%)"] = (
        (corr_final["Monitoring Predictions"] - corr_final["Reference Predictions"])
        / corr_final["Reference Predictions"].abs()
        * 100
    ).round(2)

    # Add Pass/Fail column based on absolute drift
    corr_final["Pass/Fail"] = (
        corr_final["Drift (%)"]
        .abs()
        .apply(lambda x: "Pass" if x < drift_pct_threshold else "Fail")
    )

    # Create plotly figure
    fig = go.Figure()

    # Add reference predictions bar
    fig.add_trace(
        go.Bar(
            name="Reference Prediction Correlation",
            x=corr_final.index,
            y=corr_final["Reference Predictions"],
            marker_color="blue",
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.75,
        )
    )

    # Add monitoring predictions bar
    fig.add_trace(
        go.Bar(
            name="Monitoring Prediction Correlation",
            x=corr_final.index,
            y=corr_final["Monitoring Predictions"],
            marker_color="green",
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.75,
        )
    )

    # Update layout
    fig.update_layout(
        title="Correlation between Predictions and Features",
        xaxis_title="Features",
        yaxis_title="Correlation",
        barmode="group",
        template="plotly_white",
        showlegend=True,
        xaxis_tickangle=-45,
        yaxis=dict(
            range=[-1, 1],  # Correlation range is always -1 to 1
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="grey",
            gridcolor="lightgrey",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )

    # Ensure Features is the first column
    corr_final["Feature"] = corr_final.index
    cols = ["Feature"] + [col for col in corr_final.columns if col != "Feature"]
    corr_final = corr_final[cols]

    # Calculate overall pass/fail
    pass_fail_bool = (corr_final["Pass/Fail"] == "Pass").all()

    return (
        {"Correlation Pair Table": corr_final},
        fig,
        pass_fail_bool,
        RawData(
            reference_correlations=corr_ref.to_dict(),
            monitoring_correlations=corr_mon.to_dict(),
            model=model.input_id,
            dataset_reference=datasets[0].input_id,
            dataset_monitoring=datasets[1].input_id,
        ),
    )
