# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMDataset, VMModel

try:
    from scipy.stats import kurtosis, skew
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for TargetPredictionDistributionPlot. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e


@tags("visualization")
@tasks("monitoring")
def TargetPredictionDistributionPlot(
    datasets: List[VMDataset],
    model: VMModel,
    drift_pct_threshold: float = 20,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
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

    # Get predictions
    pred_ref = datasets[0].y_prob_df(model)
    pred_ref.columns = ["Reference Prediction"]
    pred_monitor = datasets[1].y_prob_df(model)
    pred_monitor.columns = ["Monitoring Prediction"]

    # Calculate distribution moments
    moments = pd.DataFrame(
        {
            "Statistic": ["Mean", "Std", "Skewness", "Kurtosis"],
            "Reference": [
                pred_ref["Reference Prediction"].mean(),
                pred_ref["Reference Prediction"].std(),
                skew(pred_ref["Reference Prediction"]),
                kurtosis(pred_ref["Reference Prediction"]),
            ],
            "Monitoring": [
                pred_monitor["Monitoring Prediction"].mean(),
                pred_monitor["Monitoring Prediction"].std(),
                skew(pred_monitor["Monitoring Prediction"]),
                kurtosis(pred_monitor["Monitoring Prediction"]),
            ],
        }
    )

    # Calculate drift percentage with direction
    moments["Drift (%)"] = (
        (moments["Monitoring"] - moments["Reference"])
        / moments["Reference"].abs()
        * 100
    ).round(2)

    # Add Pass/Fail column based on absolute drift
    moments["Pass/Fail"] = (
        moments["Drift (%)"]
        .abs()
        .apply(lambda x: "Pass" if x < drift_pct_threshold else "Fail")
    )

    # Set Statistic as index but keep it as a column
    moments = moments.set_index("Statistic", drop=False)

    # Create KDE for both distributions
    ref_kde = ff.create_distplot(
        [pred_ref["Reference Prediction"].values],
        ["Reference"],
        show_hist=False,
        show_rug=False,
    )
    monitor_kde = ff.create_distplot(
        [pred_monitor["Monitoring Prediction"].values],
        ["Monitoring"],
        show_hist=False,
        show_rug=False,
    )

    # Create new figure
    fig = go.Figure()

    # Add reference distribution
    fig.add_trace(
        go.Scatter(
            x=ref_kde.data[0].x,
            y=ref_kde.data[0].y,
            fill="tozeroy",
            name="Reference Prediction",
            line=dict(color="blue", width=2),
            opacity=0.6,
        )
    )

    # Add monitoring distribution
    fig.add_trace(
        go.Scatter(
            x=monitor_kde.data[0].x,
            y=monitor_kde.data[0].y,
            fill="tozeroy",
            name="Monitor Prediction",
            line=dict(color="red", width=2),
            opacity=0.6,
        )
    )

    # Update layout
    fig.update_layout(
        title="Distribution of Reference & Monitor Predictions",
        xaxis_title="Prediction",
        yaxis_title="Density",
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
    )

    pass_fail_bool = (moments["Pass/Fail"] == "Pass").all()

    return (
        {"Distribution Moments": moments},
        fig,
        pass_fail_bool,
        RawData(
            pred_ref=pred_ref,
            pred_monitor=pred_monitor,
            model=model.input_id,
            dataset_reference=datasets[0].input_id,
            dataset_monitoring=datasets[1].input_id,
        ),
    )
