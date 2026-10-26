# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

from ._segments import MISSING, mean_and_half_width, segment_columns_or_default, z_value


@tags("model_performance", "stability", "time_series_data", "visualization")
@tasks("regression")
def PerformanceOverTime(
    dataset: VMDataset,
    model: VMModel,
    time_column: str,
    min_period_size: int = 30,
    confidence_level: float = 0.95,
    adjust_for_multiple_periods: bool = True,
) -> Tuple[pd.DataFrame, go.Figure, bool, RawData]:
    """
    Tracks a regression model's bias and accuracy period by period, to show whether and
    when its performance changes over time.

    ### Purpose

    A model validated on one period is used on the next. Customer mix, behaviour and the
    economic environment change, and a model can degrade gradually or break at a point.
    Out-of-time performance as a single number says *whether* it degraded; performance per
    period says *when* and *how* — a step change points to an event, a trend to drift —
    which is what an ongoing-monitoring plan and a recalibration decision need.

    ### Test Mechanism

    The dataset is grouped by `time_column` (any sortable period label, such as `2025Q3`
    or a date), in sorted order. For each period the test reports the count, the mean
    actual and predicted values, the mean residual (`actual - predicted`) with its
    confidence interval, the MAE and the R². A period is **flagged** when its bias
    interval excludes zero; with `adjust_for_multiple_periods` (default) the confidence
    level is Bonferroni-adjusted for the number of periods assessed. Periods smaller than
    `min_period_size` are reported but not assessed. Rows with no period are reported
    together as a final `Missing` period.

    The plot shows the bias with its interval and the MAE over time.

    The test passes when no period is flagged.

    ### Signs of High Risk

    - Consecutive flagged periods, especially the most recent ones.
    - A step change in bias or MAE: a structural break the model does not capture.
    - A steady trend in bias: gradual drift that will keep growing.

    ### Strengths

    - Works for any regression model with predictions assigned to the dataset.
    - Distinguishes a one-off bad period from a persistent shift.

    ### Limitations

    - Needs enough observations per period to be informative.
    - Shows that performance changed, not why; pair it with feature drift (for example PSI)
    to separate a change in the population from a change in behaviour.
    """
    segment_columns_or_default(dataset, [time_column])  # validates the column
    # rows without a period are reported together, after the last period
    missing = dataset.df[time_column].isna().to_numpy()
    periods = dataset.df[time_column].to_numpy(dtype=object)
    periods[missing] = MISSING
    actual = np.asarray(dataset.y, dtype=float)
    predicted = np.asarray(dataset.y_pred(model), dtype=float)
    residuals = actual - predicted

    ordered = sorted(pd.unique(periods[~missing])) + (
        [MISSING] if missing.any() else []
    )
    sizes = {p: int((periods == p).sum()) for p in ordered}
    assessed = sum(n >= min_period_size for n in sizes.values())
    z = z_value(confidence_level, assessed if adjust_for_multiple_periods else 1)

    rows = []
    for period in ordered:
        mask = periods == period
        bias, half_width = mean_and_half_width(residuals[mask], z)
        total = ((actual[mask] - actual[mask].mean()) ** 2).sum()
        assessable = sizes[period] >= min_period_size
        rows.append(
            {
                "Period": str(period),
                "Count": sizes[period],
                "Mean Actual": float(actual[mask].mean()),
                "Mean Predicted": float(predicted[mask].mean()),
                "Mean Residual": bias,
                "CI Lower": bias - half_width,
                "CI Upper": bias + half_width,
                "MAE": float(np.abs(residuals[mask]).mean()),
                "R-Squared": (
                    float(1 - (residuals[mask] ** 2).sum() / total) if total else np.nan
                ),
                "Result": (
                    "Too small to assess"
                    if not assessable
                    else ("Biased" if abs(bias) > half_width else "OK")
                ),
            }
        )
    table = pd.DataFrame(rows)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Mean residual (positive = under-prediction)", "MAE"],
    )
    fig.add_trace(
        go.Scatter(
            x=list(table["Period"]) + list(table["Period"][::-1]),
            y=list(table["CI Upper"]) + list(table["CI Lower"][::-1]),
            fill="toself",
            fillcolor="rgba(57, 73, 171, 0.15)",
            line=dict(width=0),
            name="Confidence interval",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=table["Period"],
            y=table["Mean Residual"],
            mode="lines+markers",
            marker=dict(
                size=8,
                color=np.where(table["Result"] == "Biased", "#c62828", "#3949ab"),
            ),
            line=dict(color="#3949ab"),
            name="Mean residual",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#757575", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=table["Period"],
            y=table["MAE"],
            mode="lines+markers",
            line=dict(color="#d81b60"),
            name="MAE",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"Performance Over Time — {model.input_id} on {dataset.input_id}",
        height=600,
    )

    passed = bool((table["Result"] != "Biased").all())

    return (
        table.round(4),
        fig,
        passed,
        RawData(residuals=residuals, model=model.input_id, dataset=dataset.input_id),
    )
