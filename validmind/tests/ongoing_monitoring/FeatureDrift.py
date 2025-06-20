# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset


def calculate_psi_score(actual, expected):
    """Calculate PSI score for a single bucket."""
    return (actual - expected) * np.log((actual + 1e-6) / (expected + 1e-6))


def calculate_feature_distributions(
    reference_data, monitoring_data, feature_columns, bins
):
    """Calculate population distributions for each feature."""
    # Calculate quantiles from reference data
    quantiles = reference_data[feature_columns].quantile(
        bins, method="single", interpolation="nearest"
    )

    distributions = {}
    for dataset_name, data in [
        ("reference", reference_data),
        ("monitoring", monitoring_data),
    ]:
        for feature in feature_columns:
            for bin_idx, threshold in enumerate(quantiles[feature]):
                if bin_idx == 0:
                    mask = data[feature] < threshold
                else:
                    prev_threshold = quantiles[feature][bins[bin_idx - 1]]
                    mask = (data[feature] >= prev_threshold) & (
                        data[feature] < threshold
                    )

                count = mask.sum()
                proportion = count / len(data)
                distributions[(dataset_name, feature, bins[bin_idx])] = proportion

    return distributions


def create_distribution_plot(feature_name, reference_dist, monitoring_dist, bins):
    """Create population distribution plot for a feature."""
    fig = go.Figure()

    # Add reference distribution
    fig.add_trace(
        go.Bar(
            x=list(range(len(bins))),
            y=reference_dist,
            name="Reference",
            marker_color="blue",
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.75,
        )
    )

    # Add monitoring distribution
    fig.add_trace(
        go.Bar(
            x=list(range(len(bins))),
            y=monitoring_dist,
            name="Monitoring",
            marker_color="green",
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.75,
        )
    )

    fig.update_layout(
        title=f"Population Distribution: {feature_name}",
        xaxis_title="Bin",
        yaxis_title="Population %",
        barmode="group",
        template="plotly_white",
        showlegend=True,
        width=800,
        height=400,
    )

    return fig


@tags("visualization")
@tasks("monitoring")
def FeatureDrift(
    datasets: List[VMDataset],
    bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    feature_columns=None,
    psi_threshold=0.2,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
    """
    Evaluates changes in feature distribution over time to identify potential model drift.

    ### Purpose

    The Feature Drift test aims to evaluate how much the distribution of features has shifted over time between two
    datasets, typically training and monitoring datasets. It uses the Population Stability Index (PSI) to quantify this
    change, providing insights into the model’s robustness and the necessity for retraining or feature engineering.

    ### Test Mechanism

    This test calculates the PSI by:

    - Bucketing the distributions of each feature in both datasets.
    - Comparing the percentage of observations in each bucket between the two datasets.
    - Aggregating the differences across all buckets for each feature to produce the PSI score for that feature.

    The PSI score is interpreted as:

    - PSI < 0.1: No significant population change.
    - PSI < 0.2: Moderate population change.
    - PSI >= 0.2: Significant population change.

    ### Signs of High Risk

    - PSI >= 0.2 for any feature, indicating a significant distribution shift.
    - Consistently high PSI scores across multiple features.
    - Sudden spikes in PSI in recent monitoring data compared to historical data.

    ### Strengths

    - Provides a quantitative measure of feature distribution changes.
    - Easily interpretable thresholds for decision-making.
    - Helps in early detection of data drift, prompting timely interventions.

    ### Limitations

    - May not capture more intricate changes in data distribution nuances.
    - Assumes that bucket thresholds (quantiles) adequately represent distribution shifts.
    - PSI score interpretation can be overly simplistic for complex datasets.
    """

    # Get feature columns
    feature_columns = feature_columns or datasets[0].feature_columns

    # Get data
    reference_data = datasets[0].df
    monitoring_data = datasets[1].df

    # Calculate distributions
    distributions = calculate_feature_distributions(
        reference_data, monitoring_data, feature_columns, bins
    )

    # Calculate PSI scores
    psi_scores = {}
    for feature in feature_columns:
        psi = 0
        for bin_val in bins:
            reference_prop = distributions[("reference", feature, bin_val)]
            monitoring_prop = distributions[("monitoring", feature, bin_val)]
            psi += calculate_psi_score(monitoring_prop, reference_prop)
        psi_scores[feature] = psi

    # Create PSI score dataframe
    psi_df = pd.DataFrame(list(psi_scores.items()), columns=["Feature", "PSI Score"])

    # Add Pass/Fail column
    psi_df["Pass/Fail"] = psi_df["PSI Score"].apply(
        lambda x: "Pass" if x < psi_threshold else "Fail"
    )

    # Sort by PSI Score
    psi_df.sort_values(by=["PSI Score"], inplace=True, ascending=False)

    # Create distribution plots
    figures = []
    for feature in feature_columns:
        reference_dist = [distributions[("reference", feature, b)] for b in bins]
        monitoring_dist = [distributions[("monitoring", feature, b)] for b in bins]
        fig = create_distribution_plot(feature, reference_dist, monitoring_dist, bins)
        figures.append(fig)

    # Calculate overall pass/fail
    pass_fail_bool = (psi_df["Pass/Fail"] == "Pass").all()

    # Prepare raw data
    raw_data = RawData(
        distributions=distributions,
        dataset_reference=datasets[0].input_id,
        dataset_monitoring=datasets[1].input_id,
    )

    return ({"PSI Scores": psi_df}, *figures, pass_fail_bool, raw_data)
