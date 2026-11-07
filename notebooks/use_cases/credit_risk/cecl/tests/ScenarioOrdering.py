# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
import plotly.graph_objects as go

from cecl_model import segments
from validmind import tags, tasks
from validmind.vm_models import VMDataset

COLORS = {"baseline": "#083E44", "adverse": "#F2A541", "severely_adverse": "#DE257E"}


@tags("cecl", "credit_risk", "scenario_analysis", "sanity_check")
@tasks("regression")
def ScenarioOrdering(
    dataset: VMDataset,
    scenarios: list = None,
    min_balance_share: float = 0.01,
):
    """
    Checks that expected credit loss rises with scenario severity, overall and in every segment.

    ### Purpose

    Each scenario is a stated view of the economy, and the allowance weights them. If a more
    severe scenario produced a **lower** loss for the portfolio or one of its segments, either the
    model responds to a driver the wrong way or a second-order effect dominates. A rate cut in
    the severe scenario speeding prepayments enough to shorten lifetimes is a real example. Either
    way the scenario weights stop meaning what management thinks they mean.

    ### Test Mechanism

    The dataset is the reporting-date portfolio with lifetime ECL per scenario in columns
    `ecl_<scenario>`. For the whole portfolio and each segment (vintage, FICO band, current-LTV
    band, occupancy) the test totals ECL per scenario and checks it is non-decreasing in the order
    given by `scenarios`. Segments below `min_balance_share` of the balance are shown but not
    judged. The share of individual loans whose ordering is reversed is reported for context. The
    test passes when no judged segment is out of order.

    ### Signs of High Risk

    - Any segment where severely adverse loss is below baseline
    - A high share of loans with reversed ordering, even when segment totals are in order:
      offsetting effects are hiding inside the aggregate
    - Reversals concentrated in high-coupon loans, which points to the prepayment response to
      falling rates

    ### Strengths

    - A direct check of the property scenario weighting relies on
    - Needs no re-run of the model: it reads the allowance that was booked

    ### Limitations

    - Ordering is necessary, not sufficient: it says nothing about whether the gaps between
      scenarios are the right size
    - Loan-level reversals can be legitimate and are reported, not judged
    """
    scenarios = scenarios or ["baseline", "adverse", "severely_adverse"]
    df = dataset.df
    cols = [f"ecl_{s}" for s in scenarios]
    total_balance = df["balance"].sum()

    keys = {"portfolio": pd.Series("all", index=df.index), **segments(df)}
    rows = []
    for segment, key in keys.items():
        g = df.groupby(key)
        t = g[cols].sum()
        t.columns = scenarios
        t["balance_share"] = g["balance"].sum() / total_balance
        t["in_order"] = (t[scenarios].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
        t["judged"] = t["balance_share"] >= min_balance_share
        t.insert(0, "segment", segment)
        rows.append(t.reset_index(names="value"))
    table = pd.concat(rows, ignore_index=True)
    table["value"] = table["value"].astype(str)

    loan_reversed = (df[cols].diff(axis=1).iloc[:, 1:] < 0).any(axis=1)
    loan_summary = pd.DataFrame(
        {
            "loans": [len(df)],
            "loans_with_reversed_ordering": [int(loan_reversed.sum())],
            "share_of_balance": [
                df.loc[loan_reversed, "balance"].sum() / total_balance
            ],
        }
    )

    vintage = table[table["segment"] == "vintage"]
    balance = df.groupby(keys["vintage"])["balance"].sum()
    fig = go.Figure()
    for s in scenarios:
        fig.add_trace(
            go.Scatter(
                x=vintage["value"],
                y=vintage[s].to_numpy() / balance.to_numpy() * 100,
                name=s,
                line_color=COLORS.get(s),
            )
        )
    fig.update_layout(
        title="Lifetime ECL / balance by vintage and scenario",
        xaxis_title="vintage",
        yaxis_title="% of balance",
        height=400,
    )

    judged = table[table["judged"]]
    return (
        {
            "ECL by scenario and segment": table.round(4),
            "Loan-level ordering": loan_summary.round(4),
        },
        fig,
        bool(judged["in_order"].all()),
    )
