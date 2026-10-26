# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("cecl", "credit_risk", "backtesting", "vintage")
@tasks("classification")
def VintageBacktest(
    dataset: VMDataset,
    model: VMModel,
    target_column: str = None,
    z_threshold: float = 2.0,
    min_expected: float = 5.0,
):
    """
    Back-tests realized against predicted events by origination vintage.

    ### Purpose

    Underwriting standards change over time in ways a loan tape does not record: documentation
    requirements, layered risk, broker channels. When they do, every loan of the affected vintages
    carries risk the model cannot see, and the model blends it across all vintages, under-reserving
    the weak ones and over-reserving the strong ones. A vintage back-test exposes this because the
    bias lines up with origination year rather than with calendar time.

    ### Test Mechanism

    The test groups loan-quarters by origination year and sums the model's predicted probabilities
    to get each vintage's expected number of events, then compares it with the realized count. The
    actual-to-expected ratio (A/E) measures the size of the bias and a z-score,
    $(A - E) / \\sqrt{\\sum p(1-p)}$, its significance. A vintage is **flagged** when
    $|z| >$ `z_threshold` and its expected count is at least `min_expected`. The test passes when no
    vintage is flagged.

    ### Signs of High Risk

    - A run of adjacent vintages with A/E well above 1, the signature of an unobserved underwriting
      shift
    - Recent vintages with A/E above 1: they dominate the current portfolio, so the bias flows
      straight into the allowance
    - A/E trending with origination year, which suggests a vintage effect should be modelled

    ### Strengths

    - Separates vintage effects from calendar-time effects, which the quarterly calibration test
      cannot do
    - Simple to read and to explain to auditors: realized events against what the model expected
      for the same loans and periods

    ### Limitations

    - Young vintages have few events, so their A/E is noisy; `min_expected` keeps them from being
      flagged on noise alone
    - A vintage effect can be confounded with the economic conditions every loan of that vintage
      went through
    """
    df = dataset.df
    target = target_column or dataset.target_column
    p = dataset.y_prob(model)
    frame = pd.DataFrame(
        {
            "vintage": df["orig_quarter"].astype(str).str[:4].astype(int),
            "loan_id": df["loan_id"],
            "actual": df[target].to_numpy(),
            "p": p,
            "var": p * (1 - p),
        }
    )
    g = frame.groupby("vintage")
    table = pd.DataFrame(
        {
            "loans": g["loan_id"].nunique(),
            "loan_quarters": g.size(),
            "actual_events": g["actual"].sum(),
            "expected_events": g["p"].sum(),
        }
    )
    table["actual_to_expected"] = table["actual_events"] / table["expected_events"]
    table["z_score"] = (table["actual_events"] - table["expected_events"]) / np.sqrt(
        g["var"].sum()
    )
    table["flagged"] = (table["z_score"].abs() > z_threshold) & (
        table["expected_events"] >= min_expected
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=table.index,
            y=table["actual_events"],
            name="actual",
            marker_color="#083E44",
        )
    )
    fig.add_trace(
        go.Bar(
            x=table.index,
            y=table["expected_events"],
            name="expected",
            marker_color="#DE257E",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table.index,
            y=table["actual_to_expected"],
            name="A/E",
            mode="lines+markers",
            line_color="#F2A541",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=1, line_dash="dash", line_color="grey", secondary_y=True)
    fig.update_layout(
        title=f"{target}: actual vs expected by vintage ({dataset.input_id})",
        barmode="group",
        height=420,
    )
    fig.update_yaxes(title_text="events", secondary_y=False)
    fig.update_yaxes(title_text="actual / expected", secondary_y=True)

    return (
        {"Back-test by vintage": table.reset_index().round(4)},
        fig,
        not bool(table["flagged"].any()),
    )
