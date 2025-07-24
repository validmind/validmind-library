# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import sys
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.logging import get_logger
from validmind.vm_models import VMDataset, VMModel

try:
    from fairlearn.metrics import (
        MetricFrame,
        count,
        demographic_parity_ratio,
        equalized_odds_ratio,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )
except ImportError as e:
    raise MissingDependencyError(
        "Missing required package `fairlearn` for ProtectedClassesCombination.",
        required_dependencies=["fairlearn"],
    ) from e

logger = get_logger(__name__)


@tags("bias_and_fairness")
@tasks("classification", "regression")
def ProtectedClassesCombination(
    dataset: VMDataset, model: VMModel, protected_classes=None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], go.Figure, RawData]:
    """
    Visualizes combinations of protected classes and their corresponding error metric differences.

    ### Purpose

    This test aims to provide insights into how different combinations of protected classes affect various error metrics,
    particularly the false negative rate (FNR) and false positive rate (FPR). By visualizing these combinations,
    it helps identify potential biases or disparities in model performance across different intersectional groups.

    ### Test Mechanism

    The test performs the following steps:
    1. Combines the specified protected class columns to create a single multi-class category.
    2. Calculates error metrics (FNR, FPR, etc.) for each combination of protected classes.
    3. Generates visualizations showing the distribution of these metrics across all class combinations.

    ### Signs of High Risk

    - Large disparities in FNR or FPR across different protected class combinations.
    - Consistent patterns of higher error rates for specific combinations of protected attributes.
    - Unexpected or unexplainable variations in error metrics between similar group combinations.

    ### Strengths

    - Provides a comprehensive view of intersectional fairness across multiple protected attributes.
    - Allows for easy identification of potentially problematic combinations of protected classes.
    - Visualizations make it easier to spot patterns or outliers in model performance across groups.

    ### Limitations

    - May become complex and difficult to interpret with a large number of protected classes or combinations.
    - Does not provide statistical significance of observed differences.
    - Visualization alone may not capture all nuances of intersectional fairness.
    """

    if sys.version_info < (3, 9):
        raise RuntimeError("This test requires Python 3.9 or higher.")

    if protected_classes is None:
        logger.warning(
            "No protected classes provided. Please pass the 'protected_classes' parameter to run this test."
        )
        return pd.DataFrame()

    # Construct a function dictionary for figures
    my_metrics = {
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "selection rate": selection_rate,
        "count": count,
    }

    # Construct a MetricFrame for figures
    mf = MetricFrame(
        metrics=my_metrics,
        y_true=dataset.y,
        y_pred=dataset.y_pred(model),
        sensitive_features=dataset._df[protected_classes],
    )

    # Combine protected class columns to create a single multi-class category for the x-axis
    metrics_by_group = mf.by_group.reset_index()
    metrics_by_group["class_combination"] = metrics_by_group[protected_classes].apply(
        lambda row: ", ".join(row.values.astype(str)), axis=1
    )

    # Create the subplots for the bar plots
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "False Positive Rate",
            "True Positive Rate",
            "Selection Rate",
            "Count",
        ],
    )

    # Add bar plots for each metric
    fig.add_trace(
        go.Bar(
            x=metrics_by_group["class_combination"],
            y=metrics_by_group["fpr"],
            name="FPR",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=metrics_by_group["class_combination"],
            y=metrics_by_group["tpr"],
            name="TPR",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=metrics_by_group["class_combination"],
            y=metrics_by_group["selection rate"],
            name="Selection Rate",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=metrics_by_group["class_combination"],
            y=metrics_by_group["count"],
            name="Count",
        ),
        row=2,
        col=2,
    )

    # Update layout of the figure to match the original style
    fig.update_layout(
        title="Show all metrics",
        height=800,
        width=900,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(t=50),
        font=dict(size=12),
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=2)

    # Extract demographic parity ratio and equalized odds ratio
    m_dpr = []
    m_eqo = []
    for protected_class in protected_classes:
        m_dpr.append(
            demographic_parity_ratio(
                y_true=dataset.y,
                y_pred=dataset.y_pred(model),
                sensitive_features=dataset._df[[protected_class]],
            )
        )
        m_eqo.append(
            equalized_odds_ratio(
                y_true=dataset.y,
                y_pred=dataset.y_pred(model),
                sensitive_features=dataset._df[[protected_class]],
            )
        )

    # Create a DataFrame for the demographic parity and equalized odds ratio
    dpr_eor_df = pd.DataFrame(
        columns=protected_classes,
        index=["demographic parity ratio", "equal odds ratio"],
    )

    for i in range(len(m_dpr)):
        dpr_eor_df[protected_classes[i]]["demographic parity ratio"] = round(
            m_dpr[i], 2
        )
        dpr_eor_df[protected_classes[i]]["equal odds ratio"] = round(m_eqo[i], 2)

    return (
        {"Class Combination Table": metrics_by_group},
        {"DPR and EOR table": dpr_eor_df},
        fig,
        RawData(
            metrics_frame=mf,
            demographic_parity_ratios=m_dpr,
            equalized_odds_ratios=m_eqo,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
