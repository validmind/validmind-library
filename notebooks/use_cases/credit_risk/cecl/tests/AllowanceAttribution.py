# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
import plotly.graph_objects as go

from cecl_model import (
    RS_QUARTERS,
    REVERSION_QUARTERS,
    SCENARIO_WEIGHTS,
    build_scenario_paths,
    segments,
    weighted_allowance,
)
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

PERIOD_COLORS = ["#083E44", "#F2A541", "#6FA8B0"]


@tags("cecl", "credit_risk", "allowance", "concentration")
@tasks("regression")
def AllowanceAttribution(
    dataset: VMDataset,
    models: list[VMModel],
    rs_quarters: int = RS_QUARTERS,
    reversion_quarters: int = REVERSION_QUARTERS,
    concentration_ratio: float = 2.0,
    min_ecl_share: float = 0.05,
):
    """
    Breaks the allowance down by where in the loans' lives it falls and by portfolio segment.

    ### Purpose

    A single allowance figure hides what drives it. Two questions matter for CECL governance.
    The first is how much of the loss comes from the reasonable and supportable forecast, the part
    that reflects management's economic view, and how much from reversion and the remaining life
    at long-run conditions. The second is which segments carry the loss relative to their size. A
    segment holding far more of the allowance than of the balance is a concentration that needs
    its own scrutiny and is the first place a model error would show.

    ### Test Mechanism

    The dataset is the reporting-date portfolio and `models` are the default, prepayment and LGD
    models, in that order. The test re-runs the lifetime engine per scenario and splits each
    loan's expected loss by forecast quarter into the forecast period (`rs_quarters`), the
    reversion period (`reversion_quarters`) and the remaining life, per scenario and weighted.
    It then totals the weighted allowance by segment (vintage, FICO band, current-LTV band,
    occupancy) against each segment's share of balance. A segment is **flagged** when its share of
    the allowance is at least `concentration_ratio` times its share of balance and at least
    `min_ecl_share` of the total. The test is informational and reports flags rather than
    failing.

    ### Signs of High Risk

    - Most of the allowance sitting in the remaining-life period, so long-run assumptions rather
      than the economic forecast drive the number
    - A flagged segment that is also growing, such as recent vintages or high-LTV loans
    - A severe scenario that moves only the forecast period, which means the shock does not
      reach the long-dated losses

    ### Strengths

    - Connects the allowance to the two judgments that drive it (forecast horizon and portfolio
      mix) in one view
    - Gives reviewers a ranked list of where to look first

    ### Limitations

    - Segments are one-dimensional; concentrations that only appear in combination (for example
      low FICO and high LTV) need a cross-tab
    """
    portfolio = dataset.df
    raw = tuple(m.model for m in models)
    paths = build_scenario_paths(rs_quarters, reversion_quarters)
    _, results = weighted_allowance(portfolio, paths, raw)

    cut = rs_quarters + reversion_quarters
    periods = {
        "reasonable_and_supportable": slice(0, rs_quarters),
        "reversion": slice(rs_quarters, cut),
        "remaining_life": slice(cut, None),
    }
    by_period = pd.DataFrame(
        {
            s: {p: r["loss"][:, sl].sum() for p, sl in periods.items()}
            for s, r in results.items()
        }
    )
    by_period["weighted"] = sum(w * by_period[s] for s, w in SCENARIO_WEIGHTS.items())
    by_period.loc["total"] = by_period.sum()
    shares = by_period / by_period.loc["total"]

    weighted_ecl = sum(w * results[s]["ecl"] for s, w in SCENARIO_WEIGHTS.items())
    frame = portfolio.assign(weighted_ecl=weighted_ecl)
    total_ecl, total_balance = weighted_ecl.sum(), portfolio["balance"].sum()
    rows = []
    for segment, key in segments(frame).items():
        g = frame.groupby(key)
        t = pd.DataFrame(
            {
                "loans": g.size(),
                "balance_share": g["balance"].sum() / total_balance,
                "ecl_share": g["weighted_ecl"].sum() / total_ecl,
                "allowance_ratio": g["weighted_ecl"].sum() / g["balance"].sum(),
            }
        )
        t["ecl_to_balance_share"] = t["ecl_share"] / t["balance_share"]
        t["flagged"] = (t["ecl_to_balance_share"] >= concentration_ratio) & (
            t["ecl_share"] >= min_ecl_share
        )
        t.insert(0, "segment", segment)
        rows.append(t.reset_index(names="value"))
    seg = pd.concat(rows, ignore_index=True)
    seg["value"] = seg["value"].astype(str)

    fig = go.Figure()
    for p, color in zip(periods, PERIOD_COLORS):
        fig.add_trace(
            go.Bar(
                x=by_period.columns,
                y=by_period.loc[p] / 1e6,
                name=p,
                marker_color=color,
            )
        )
    fig.update_layout(
        title="Lifetime expected loss by period of the loans' lives",
        barmode="stack",
        yaxis_title="$m",
        height=400,
    )

    return (
        {
            "Expected loss by period ($)": by_period.reset_index(names="period").round(
                0
            ),
            "Share of expected loss by period": shares.reset_index(
                names="period"
            ).round(4),
            "Allowance by segment": seg.round(4),
        },
        fig,
    )
