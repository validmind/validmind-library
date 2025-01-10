# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
import plotly.graph_objs as go
from typing import List, Tuple
from validmind import tags, tasks
from validmind.vm_models import VMDataset
from validmind.errors import SkipTestError


@tags("tabular_data", "binary_classification", "multiclass_classification")
@tasks("classification")
def ClassImbalanceDrift(
    datasets: List[VMDataset],
    drift_pct_threshold: float = 5.0,
    title: str = "Class Distribution Drift",
) -> Tuple[go.Figure, dict, bool]:
    """
    Evaluates drift in class distribution between reference and monitoring datasets.

    ### Purpose
    This test compares the class distribution between two datasets to identify
    potential population drift in the target variable.

    ### Test Mechanism
    - Calculates class percentages for both datasets
    - Computes drift as the difference in percentages
    - Visualizes distributions side by side
    - Flags significant changes in class proportions

    ### Signs of High Risk
    - Large shifts in class proportions
    - New classes appearing or existing classes disappearing
    - Multiple classes showing significant drift
    - Systematic shifts across multiple classes
    """
    # Validate inputs
    if not datasets[0].target_column or not datasets[1].target_column:
        raise SkipTestError("No target column provided")

    # Calculate class distributions
    ref_dist = (
        datasets[0].df[datasets[0].target_column].value_counts(normalize=True) * 100
    )
    mon_dist = (
        datasets[1].df[datasets[1].target_column].value_counts(normalize=True) * 100
    )

    # Get all unique classes
    all_classes = sorted(set(ref_dist.index) | set(mon_dist.index))

    if len(all_classes) > 10:
        raise SkipTestError("Skipping target column with more than 10 classes")

    # Create comparison table
    rows = []
    all_passed = True

    for class_label in all_classes:
        ref_percent = ref_dist.get(class_label, 0)
        mon_percent = mon_dist.get(class_label, 0)

        # Calculate drift (preserving sign)
        drift = mon_percent - ref_percent
        passed = abs(drift) < drift_pct_threshold
        all_passed &= passed

        rows.append(
            {
                datasets[0].target_column: class_label,
                "Reference (%)": round(ref_percent, 4),
                "Monitoring (%)": round(mon_percent, 4),
                "Drift (%)": round(drift, 4),
                "Pass/Fail": "Pass" if passed else "Fail",
            }
        )

    comparison_df = pd.DataFrame(rows)

    # Create named tables dictionary
    tables = {"Class Distribution (%)": comparison_df}

    # Create visualization
    fig = go.Figure()

    # Add reference distribution bar
    fig.add_trace(
        go.Bar(
            name="Reference",
            x=[str(c) for c in all_classes],
            y=comparison_df["Reference (%)"],
            marker_color="rgba(31, 119, 180, 0.8)",  # Blue with 0.8 opacity
        )
    )

    # Add monitoring distribution bar
    fig.add_trace(
        go.Bar(
            name="Monitoring",
            x=[str(c) for c in all_classes],
            y=comparison_df["Monitoring (%)"],
            marker_color="rgba(255, 127, 14, 0.8)",  # Orange with 0.8 opacity
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Percentage (%)",
        barmode="group",
        showlegend=True,
    )

    return fig, tables, all_passed
