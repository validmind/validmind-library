# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
import plotly.graph_objects as go

from cecl_model import (
    RS_QUARTERS,
    REVERSION_QUARTERS,
    build_scenario_paths,
    weighted_allowance,
)
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("cecl", "credit_risk", "sensitivity", "assumptions")
@tasks("regression")
def ForecastPeriodSensitivity(
    dataset: VMDataset,
    models: list[VMModel],
    rs_quarters_grid: list = None,
    reversion_quarters_grid: list = None,
    scenario_weights: dict = None,
):
    """
    Measures how the allowance changes with the length of the reasonable and supportable period and
    the reversion period.

    ### Purpose

    ASC 326 leaves two choices to management: how far ahead the economic forecast is reasonable
    and supportable, and how the estimate reverts to historical experience after that. Neither is
    estimated from data, yet both can move the allowance materially. When current conditions are
    far from the long-run average, a longer forecast period keeps them in the estimate for longer.
    Examiners expect the choice to be justified and its impact known. This test puts a number on
    it.

    ### Test Mechanism

    The dataset is the reporting-date portfolio and `models` are the default, prepayment and LGD
    models, in that order. For every combination of `rs_quarters_grid` and
    `reversion_quarters_grid` the test rebuilds the three scenario paths and re-runs the full
    lifetime engine. It then reports the probability-weighted allowance, its ratio to balance and
    its change against the model's chosen setting (8 and 4 quarters). A reversion of 0 quarters
    means immediate reversion. The test is informational and has no pass/fail threshold.

    ### Signs of High Risk

    - A large change in allowance for a modest change in either period: the estimate rests on a
      judgment that needs strong support
    - An allowance that rises steadily with the forecast period, which means the scenarios
      stay well above long-run loss levels; ask whether that is still supportable at the chosen
      cut-off
    - Results that are insensitive to every setting, which can mean the macro drivers barely
      reach the loss estimate

    ### Strengths

    - Re-runs the same engine the allowance comes from, so the sensitivity is exact rather than
      approximated
    - Covers both management judgments in one grid

    ### Limitations

    - The forecast period cannot exceed the 13 quarters the Fed scenarios cover
    - Holds scenario weights fixed; their sensitivity is a separate test
    """
    rs_grid = rs_quarters_grid or [4, 8, 12]
    rev_grid = reversion_quarters_grid or [0, 4, 8]
    portfolio = dataset.df
    raw = tuple(m.model for m in models)
    balance = portfolio["balance"].sum()

    rows = []
    for rs in rs_grid:
        for rev in rev_grid:
            paths = build_scenario_paths(rs, rev)
            total, _ = weighted_allowance(portfolio, paths, raw, scenario_weights)
            rows.append(
                {
                    "rs_quarters": rs,
                    "reversion_quarters": rev,
                    "allowance": total,
                    "allowance_ratio": total / balance,
                }
            )
    table = pd.DataFrame(rows)
    base = table[
        (table["rs_quarters"] == RS_QUARTERS)
        & (table["reversion_quarters"] == REVERSION_QUARTERS)
    ]["allowance"]
    if len(base):
        table["change_vs_chosen"] = table["allowance"] / base.iloc[0] - 1

    grid = table.pivot(
        index="rs_quarters", columns="reversion_quarters", values="allowance_ratio"
    )
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy() * 100,
            x=[f"{c} q reversion" for c in grid.columns],
            y=[f"{r} q forecast" for r in grid.index],
            text=[[f"{v * 100:.3f}%" for v in row] for row in grid.to_numpy()],
            texttemplate="%{text}",
            colorscale="Teal",
            colorbar_title="% of balance",
        )
    )
    fig.update_layout(title="Probability-weighted allowance / balance", height=380)

    return {"Allowance by forecast and reversion period": table.round(6)}, fig
