# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objs as go

from validmind import tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset


@tags("tabular_data", "binary_classification", "multiclass_classification")
@tasks("classification")
def ClassImbalanceDrift(
    datasets: List[VMDataset],
    drift_pct_threshold: float = 5.0,
    title: str = "Class Distribution Drift",
) -> Tuple[go.Figure, Dict[str, pd.DataFrame], bool]:
    """
    Evaluates drift in class distribution between reference and monitoring datasets.

    ### Purpose

    The Class Imbalance Drift test is designed to detect changes in the distribution of target classes
    over time. By comparing class proportions between reference and monitoring datasets, this test helps
    identify whether the population structure remains stable in production. This is crucial for
    understanding if the model continues to operate under similar class distribution assumptions and
    whether retraining might be necessary due to significant shifts in class balance.

    ### Test Mechanism

    This test proceeds by calculating class percentages for both reference and monitoring datasets.
    It computes the proportion of each class and quantifies drift as the percentage difference in these
    proportions between datasets. The test provides both visual and numerical comparisons of class
    distributions, with special attention to changes that exceed the specified drift threshold.
    Population stability is assessed on a class-by-class basis.

    ### Signs of High Risk

    - Large shifts in class proportions exceeding the threshold
    - Systematic changes affecting multiple classes
    - Appearance of new classes or disappearance of existing ones
    - Significant changes in minority class representation
    - Reversal of majority-minority class relationships
    - Unexpected changes in class ratios

    ### Strengths

    - Provides clear visualization of distribution changes
    - Identifies specific classes experiencing drift
    - Enables early detection of population shifts
    - Includes standardized drift threshold evaluation
    - Supports both binary and multiclass problems
    - Maintains interpretable percentage-based metrics

    ### Limitations

    - Does not account for feature distribution changes
    - Cannot identify root causes of class drift
    - May be sensitive to small sample sizes
    - Limited to target variable distribution only
    - Requires sufficient samples per class
    - May not capture subtle distribution changes
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
