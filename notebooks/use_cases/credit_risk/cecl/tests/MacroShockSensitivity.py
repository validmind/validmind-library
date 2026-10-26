# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cecl_model import (
    REVERSION_QUARTERS,
    RS_QUARTERS,
    lifetime_ecl,
    path_from_changes,
    scenario_changes,
    shocked_changes,
)
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

DEFAULT_SHOCKS = {
    "unemployment_rate": [1, 2, 3, 4, 5],
    "house_price_index": [-0.1, -0.2, -0.3, -0.4],
    "mortgage_rate": [-2, -1, 1, 2],
}
# The direction a larger shock should push the allowance. A mortgage rate rise slows
# prepayment, which lengthens loan lives, but it is not a credit driver: not judged.
EXPECTED_DIRECTION = {"unemployment_rate": 1, "house_price_index": -1}


@tags("cecl", "credit_risk", "stress_testing", "sensitivity")
@tasks("regression")
def MacroShockSensitivity(
    dataset: VMDataset,
    models: list[VMModel],
    shocks: dict = None,
    rs_quarters: int = RS_QUARTERS,
    reversion_quarters: int = REVERSION_QUARTERS,
):
    """
    Measures how the lifetime allowance responds to shocks to each macroeconomic driver, one at
    a time and combined.

    ### Purpose

    The allowance is a forecast of losses conditional on the economy. Management, examiners and
    capital planning all need to know which macro driver it depends on most, how strongly, and
    whether it responds in the direction the economics say it should. A driver that barely
    moves the allowance, or moves it the wrong way, means the stress scenarios do not reach the
    loss estimate the way they are assumed to. This test isolates each driver.

    ### Test Mechanism

    The dataset is the reporting-date portfolio and `models` are the default, prepayment and LGD
    models, in that order. Starting from the baseline scenario, each shock in `shocks` is added
    to one driver, building in a straight line to its full size by the end of the reasonable and
    supportable period: unemployment and mortgage rate in percentage points, house prices as a
    fractional change in level. Unemployment and the mortgage rate then revert to their long-run
    averages; a house-price fall is permanent. The full lifetime engine is re-run for each shock
    and once with the largest unemployment and house-price shocks together. The test passes when
    the allowance rises steadily with larger unemployment shocks and deeper house-price falls.

    ### Signs of High Risk

    - An allowance that falls, or stays flat, as unemployment rises or house prices fall
    - A response that flattens beyond some shock size: the model is extrapolating flat beyond
      the conditions it was trained on, and capping the stress
    - A combined shock far larger than the sum of the single shocks, or far smaller, which shows
      strong interactions that scenario design should account for

    ### Strengths

    - Re-runs the same engine the allowance comes from
    - Separates the drivers that the Fed scenarios move together

    ### Limitations

    - Shocks one driver while holding the others at baseline, a combination the economy rarely
      produces
    - Uses the baseline scenario only, without scenario weights
    """
    shocks = shocks or DEFAULT_SHOCKS
    portfolio = dataset.df
    raw = tuple(m.model for m in models)
    balance = portfolio["balance"].sum()
    baseline = scenario_changes(rs_quarters)["baseline"]

    def allowance(**shock):
        path = path_from_changes(
            shocked_changes(baseline, **shock), reversion_quarters
        )
        return lifetime_ecl(portfolio, path, *raw)["ecl"].sum()

    base = allowance()
    rows = [{"driver": "none", "shock": 0.0, "allowance": base}]
    for driver, sizes in shocks.items():
        for size in sizes:
            rows.append(
                {"driver": driver, "shock": size, "allowance": allowance(**{driver: size})}
            )
    combined = {
        "unemployment_rate": max(shocks["unemployment_rate"]),
        "house_price_index": min(shocks["house_price_index"]),
    }
    rows.append(
        {
            "driver": "combined",
            "shock": np.nan,
            "allowance": allowance(**combined),
            "detail": ", ".join(f"{k} {v:+g}" for k, v in combined.items()),
        }
    )
    table = pd.DataFrame(rows)
    table["allowance_ratio"] = table["allowance"] / balance
    table["change_vs_baseline"] = table["allowance"] / base - 1

    checks = []
    for driver, direction in EXPECTED_DIRECTION.items():
        # Include the unshocked baseline so the first shock must already move it.
        t = table[table["driver"].isin([driver, "none"])].sort_values("shock")
        steps = np.diff(t["allowance"].to_numpy()) * direction
        checks.append(
            {
                "driver": driver,
                "expected": "allowance rises with stress",
                "responds_as_expected": bool((steps > 0).all()),
            }
        )
    checks = pd.DataFrame(checks)

    # One panel per driver, since the shocks are in different units.
    fig = make_subplots(rows=1, cols=len(shocks), subplot_titles=list(shocks))
    for i, driver in enumerate(shocks, start=1):
        t = table[table["driver"].isin([driver, "none"])].sort_values("shock")
        fig.add_trace(
            go.Scatter(
                x=t["shock"], y=t["allowance_ratio"] * 100, mode="lines+markers",
                line_color="#083E44",
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        title="Lifetime allowance / balance (%) under single-driver shocks, baseline scenario",
        height=380,
        showlegend=False,
    )

    return (
        {
            "Allowance by macro shock": table.round(6),
            "Direction checks": checks,
        },
        fig,
        bool(checks["responds_as_expected"].all()),
    )
