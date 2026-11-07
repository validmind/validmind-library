# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cecl_model import SCENARIO_WEIGHTS
from validmind import tags, tasks
from validmind.vm_models import VMDataset


@tags("cecl", "credit_risk", "sensitivity", "assumptions", "scenario_analysis")
@tasks("regression")
def ScenarioWeightSensitivity(
    dataset: VMDataset,
    step: float = 0.05,
    max_severe_weight: float = 0.3,
    chosen_weights: dict = None,
):
    """
    Maps the allowance across alternative scenario weightings.

    ### Purpose

    Scenario weights are a management judgment with no estimation behind them, and the gap
    between a baseline and a severe scenario is usually several-fold. A few percentage points of
    weight can therefore move the allowance more than any model refinement. This test shows the
    allowance across plausible weightings, so the chosen weights can be judged against the range
    they sit in and their impact stated plainly.

    ### Test Mechanism

    The dataset is the reporting-date portfolio with lifetime ECL per scenario in columns
    `ecl_baseline`, `ecl_adverse` and `ecl_severely_adverse`. The test steps the baseline weight
    from 0 to 1 and the severely adverse weight from 0 to `max_severe_weight` in increments of
    `step`, gives adverse the remainder, and keeps every combination where all weights are
    non-negative. For each it computes the allowance and its ratio to balance. It also reports
    the allowance range, where the chosen weights sit in it, and the allowance change per 10
    percentage points of weight moved from baseline to severely adverse. The test is informational.

    ### Signs of High Risk

    - The chosen weights sitting at the low end of a wide range without a documented reason
    - A large change per 10 points moved to the severe scenario, which means the weighting
      needs governance as strict as the model's
    - A severe scenario so far from the others that any weight on it dominates the allowance

    ### Strengths

    - Exact and cheap: the allowance is linear in the weights, so the per-scenario ECL already
      booked is enough
    - Turns a qualitative judgment into a number reviewers can challenge

    ### Limitations

    - Covers only the three scenarios run; it says nothing about scenarios that were not
      considered
    """
    chosen = chosen_weights or SCENARIO_WEIGHTS
    df = dataset.df
    ecl = {s: df[f"ecl_{s}"].sum() for s in ["baseline", "adverse", "severely_adverse"]}
    balance = df["balance"].sum()

    rows = []
    for wb in np.round(np.arange(0, 1 + 1e-9, step), 4):
        for ws in np.round(np.arange(0, max_severe_weight + 1e-9, step), 4):
            wa = round(1 - wb - ws, 4)
            if wa < 0:
                continue
            total = (
                wb * ecl["baseline"]
                + wa * ecl["adverse"]
                + ws * ecl["severely_adverse"]
            )
            rows.append(
                {
                    "baseline": wb,
                    "adverse": wa,
                    "severely_adverse": ws,
                    "allowance": total,
                    "allowance_ratio": total / balance,
                }
            )
    table = pd.DataFrame(rows)
    chosen_total = sum(w * ecl[s] for s, w in chosen.items())
    summary = pd.DataFrame(
        {
            "chosen_weights": [str(chosen)],
            "chosen_allowance": [chosen_total],
            "min_allowance": [table["allowance"].min()],
            "max_allowance": [table["allowance"].max()],
            "chosen_percentile_in_grid": [(table["allowance"] < chosen_total).mean()],
            "change_per_10pts_baseline_to_severe": [
                0.1 * (ecl["severely_adverse"] - ecl["baseline"])
            ],
            "ecl_baseline": [ecl["baseline"]],
            "ecl_adverse": [ecl["adverse"]],
            "ecl_severely_adverse": [ecl["severely_adverse"]],
        }
    )

    grid = table.pivot(
        index="severely_adverse", columns="baseline", values="allowance_ratio"
    )
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy() * 100,
            x=grid.columns,
            y=grid.index,
            colorscale="Teal",
            colorbar_title="% of balance",
            hovertemplate="baseline %{x}<br>severe %{y}<br>%{z:.3f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[chosen.get("baseline")],
            y=[chosen.get("severely_adverse")],
            mode="markers",
            marker=dict(color="#DE257E", size=12, symbol="x"),
            name="chosen",
        )
    )
    fig.update_layout(
        title="Allowance / balance by scenario weights (adverse = remainder)",
        xaxis_title="baseline weight",
        yaxis_title="severely adverse weight",
        height=420,
    )

    return (
        {"Summary": summary.round(6), "Allowance by weighting": table.round(6)},
        fig,
    )
