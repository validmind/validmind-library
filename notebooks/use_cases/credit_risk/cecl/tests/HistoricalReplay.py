# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cecl_model import (
    REVERSION_QUARTERS,
    RS_QUARTERS,
    historical_changes,
    lifetime_ecl,
    path_from_changes,
    scenario_changes,
)
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

DEFAULT_EPISODES = {
    "early_1990s_recession": "1990Q2",
    "global_financial_crisis": "2007Q4",
    "pandemic": "2019Q4",
}


@tags("cecl", "credit_risk", "stress_testing", "scenario_analysis")
@tasks("regression")
def HistoricalReplay(
    dataset: VMDataset,
    models: list[VMModel],
    episodes: dict = None,
    benchmark: str = "severely_adverse",
    rs_quarters: int = RS_QUARTERS,
    reversion_quarters: int = REVERSION_QUARTERS,
):
    """
    Replays past downturns on today's portfolio and compares the allowance with the stress
    scenario used for the estimate.

    ### Purpose

    A severe scenario is only as useful as it is severe. This test asks whether the Federal
    Reserve's severely adverse scenario, re-based to the reporting date, produces at least the
    allowance that the downturns the US actually went through would produce on the same
    portfolio. If a historical episode is worse, the scenario set understates tail risk for
    this book.

    ### Test Mechanism

    The dataset is the reporting-date portfolio and `models` are the default, prepayment and LGD
    models, in that order. For each episode in `episodes` (a name and the quarter it started
    from), the test takes the changes in unemployment, mortgage rate and house prices over the
    next `rs_quarters` quarters of real history, applies them to the reporting-quarter actuals,
    and reverts to long-run averages exactly as the Fed scenarios do. It re-runs the lifetime
    engine on each path and on each Fed scenario. The test passes when the `benchmark` scenario's
    allowance is at least that of every historical episode.

    ### Signs of High Risk

    - A historical episode producing a larger allowance than the severely adverse scenario
    - An episode with a large macro shock but a small allowance: the model may not respond to
      that kind of downturn, for example because it extrapolates flat beyond its training range
    - The pandemic replay producing large losses: the model does not know about forbearance,
      which is what held real 2020 defaults down

    ### Strengths

    - Uses real, plausible combinations of macro moves rather than hypothetical shocks
    - Re-bases history to today's starting point the same way as the Fed scenarios, so the
      comparison is like for like

    ### Limitations

    - History can only replay downturns that happened since 1990, the start of the macro data
    - Replays the macro path, not the policy response: forbearance, rate cuts and fiscal support
      are only in the path as far as they moved the three drivers
    """
    episodes = episodes or DEFAULT_EPISODES
    portfolio = dataset.df
    raw = tuple(m.model for m in models)
    balance = portfolio["balance"].sum()

    paths = {
        name: ("historical", start, historical_changes(start, rs_quarters))
        for name, start in episodes.items()
    }
    for s, changes in scenario_changes(rs_quarters).items():
        paths[s] = ("fed_2025", "", changes)

    rows = []
    for name, (source, start, changes) in paths.items():
        ecl = lifetime_ecl(
            portfolio, path_from_changes(changes, reversion_quarters), *raw
        )["ecl"].sum()
        rows.append(
            {
                "path": name,
                "source": source,
                "start": start,
                "peak_unemployment_change": changes["unemployment_rate"].max(),
                # Deepest fall in level over the period, 0 if prices only rose.
                "house_price_change": np.expm1(min(changes["hpi_log_change"].min(), 0)),
                "allowance": ecl,
                "allowance_ratio": ecl / balance,
            }
        )
    table = pd.DataFrame(rows)
    bench = table.loc[table["path"] == benchmark, "allowance"].iloc[0]
    table["relative_to_benchmark"] = table["allowance"] / bench
    historical = table[table["source"] == "historical"]
    passed = bool((historical["allowance"] <= bench).all())

    colors = {"historical": "#083E44", "fed_2025": "#DE257E"}
    t = table.sort_values("allowance_ratio")
    fig = go.Figure(
        go.Bar(
            x=t["allowance_ratio"] * 100,
            y=t["path"],
            orientation="h",
            marker_color=[colors[s] for s in t["source"]],
            text=[f"{v * 100:.3f}%" for v in t["allowance_ratio"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Lifetime allowance / balance: historical replays (dark) and Fed 2025 scenarios",
        xaxis_title="% of balance",
        height=380,
    )

    return {"Allowance by historical episode and scenario": table.round(6)}, fig, passed
