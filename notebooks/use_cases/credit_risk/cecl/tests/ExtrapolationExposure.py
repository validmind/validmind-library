# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cecl_model import (
    RS_QUARTERS,
    REVERSION_QUARTERS,
    build_scenario_paths,
    project,
)
from validmind import tags, tasks
from validmind.vm_models import VMDataset

DRIVERS = [
    "fico",
    "current_ltv",
    "dti",
    "loan_age",
    "unemployment_rate",
    "refi_incentive",
]
COLORS = {"baseline": "#083E44", "adverse": "#F2A541", "severely_adverse": "#DE257E"}


@tags("cecl", "credit_risk", "scenario_analysis", "data_coverage")
@tasks("classification", "regression")
def ExtrapolationExposure(
    datasets: list[VMDataset],
    rs_quarters: int = RS_QUARTERS,
    reversion_quarters: int = REVERSION_QUARTERS,
    max_share: float = 0.05,
):
    """
    Measures how much of the projected portfolio falls outside the range of the development data.

    ### Purpose

    A CECL projection feeds the model conditions it was never trained on: house prices after a
    shock, unemployment at a scenario peak, loans aged far beyond any loan in the sample. Where
    that happens the model is extrapolating, and a gradient-boosted model extrapolates as a flat
    line. The allowance in those quarters rests on an assumption that nothing gets worse past the
    edge of history. This test sizes that exposure per scenario and per driver.

    ### Test Mechanism

    `datasets` takes two inputs in order: the **reporting-date portfolio**, then the **development
    sample** that defines the observed range. Each loan is projected through the forecast and
    reversion quarters of every scenario, the window where scenarios differ. For each driver the
    test records the share of on-book loan-quarters whose value is below the development minimum
    or above its maximum. The test passes when no scenario and driver exceed `max_share`.

    ### Signs of High Risk

    - The severely adverse scenario pushing a macro driver (unemployment, LTV through house
      prices) outside the development range, since the model then caps the stress
    - A large out-of-range share in the first few quarters, where most of the reasonable and
      supportable loss sits
    - Loan age past the development maximum for a large part of the book, which affects the
      seasoning shape late in life

    ### Strengths

    - Ties model coverage directly to the scenarios used for the allowance, not to a generic
      hold-out sample
    - Cheap and model-agnostic: it reads only features, never predictions

    ### Limitations

    - Checks each driver on its own; combinations of in-range values can still be unseen
      jointly
    - Uses the min/max of the development sample, so a single outlier can make coverage look
      better than it is
    """
    portfolio = datasets[0].df
    development = datasets[1].df
    window = rs_quarters + reversion_quarters
    paths = build_scenario_paths(rs_quarters, reversion_quarters)
    lo, hi = development[DRIVERS].min(), development[DRIVERS].max()

    rows, by_quarter = [], {}
    for scenario, path in paths.items():
        frame, _, on_book = project(portfolio, path)
        quarter = np.tile(np.arange(on_book.shape[1]), on_book.shape[0])
        keep = on_book.ravel() & (quarter < window)
        f = frame[keep]
        outside = (f[DRIVERS] < lo) | (f[DRIVERS] > hi)
        rows.append({"scenario": scenario, **outside.mean().to_dict()})
        by_quarter[scenario] = outside.any(axis=1).groupby(quarter[keep] + 1).mean()

    table = pd.DataFrame(rows).set_index("scenario")
    ranges = pd.DataFrame({"development_min": lo, "development_max": hi})

    fig = go.Figure()
    for scenario, share in by_quarter.items():
        fig.add_trace(
            go.Scatter(
                x=share.index,
                y=share * 100,
                name=scenario,
                line_color=COLORS.get(scenario),
            )
        )
    fig.add_vline(x=rs_quarters + 0.5, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Share of on-book loan-quarters with any driver outside the development range",
        xaxis_title="forecast quarter",
        yaxis_title="% of loan-quarters",
        height=400,
    )

    return (
        {
            "Share outside development range, forecast and reversion window": table.reset_index().round(
                4
            ),
            "Development range": ranges.reset_index(names="driver").round(4),
        },
        fig,
        bool((table <= max_share).all().all()),
    )
