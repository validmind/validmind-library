# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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


@tags("cecl", "credit_risk", "stress_testing", "reverse_stress_test")
@tasks("regression")
def ReverseStress(
    dataset: VMDataset,
    models: list[VMModel],
    threshold_multiple: float = 3.0,
    full_shock: dict = None,
    severities: list = None,
    rs_quarters: int = RS_QUARTERS,
    reversion_quarters: int = REVERSION_QUARTERS,
):
    """
    Finds how severe a combined unemployment and house-price shock must be to multiply the
    baseline allowance by a given factor.

    ### Purpose

    A forward stress test asks what a chosen scenario does to the allowance. A reverse stress
    test asks the opposite: what economy would it take to reach a loss level that matters?
    Knowing the breaking point, and how it compares with the severely adverse scenario and with
    history, tells management how much room the portfolio has and whether that room depends on
    an implausibly severe downturn.

    ### Test Mechanism

    The dataset is the reporting-date portfolio and `models` are the default, prepayment and LGD
    models, in that order. `full_shock` defines a severity of 1: by default unemployment up 5
    points and house prices down 30%. Starting from the baseline scenario, the test applies the
    shock scaled by each value in `severities`, building in a straight line to its full size by
    the end of the reasonable and supportable period, and re-runs the lifetime engine. The
    breaking point is the first severity, interpolated between grid points, at which the
    allowance reaches `threshold_multiple` times the baseline allowance. The test reports the
    implied unemployment rise and house-price fall and compares them with the peak moves in the
    Fed severely adverse scenario. It is informational and has no pass/fail threshold.

    ### Signs of High Risk

    - A breaking point at or below the severity of the severely adverse scenario: the threshold
      is reached in a scenario the institution already treats as plausible
    - No breaking point within the grid: the model's response may be capped by flat
      extrapolation beyond the training range, rather than the portfolio being resilient
    - A response that steepens sharply near the breaking point, a cliff the scenario set may
      not capture

    ### Strengths

    - Turns a loss level into an economy that can be discussed and judged for plausibility
    - Re-runs the same engine the allowance comes from

    ### Limitations

    - Searches one direction only: unemployment and house prices moving together in fixed
      proportion
    - The threshold is a multiple of the allowance, not of capital or earnings, which this
      model does not see
    """
    full_shock = full_shock or {"unemployment_rate": 5.0, "house_price_index": -0.3}
    severities = severities or list(np.round(np.arange(0, 2.01, 0.2), 1))
    portfolio = dataset.df
    raw = tuple(m.model for m in models)
    balance = portfolio["balance"].sum()
    changes = scenario_changes(rs_quarters)
    baseline = changes["baseline"]

    def shock_at(severity):
        return {
            "unemployment_rate": full_shock["unemployment_rate"] * severity,
            # Scale the log change so severity 2 compounds rather than exceeds -100%.
            "house_price_index": np.expm1(
                np.log1p(full_shock["house_price_index"]) * severity
            ),
        }

    rows = []
    for severity in severities:
        shock = shock_at(severity)
        path = path_from_changes(shocked_changes(baseline, **shock), reversion_quarters)
        rows.append(
            {
                "severity": severity,
                "unemployment_shock": shock["unemployment_rate"],
                "house_price_shock": shock["house_price_index"],
                "allowance": lifetime_ecl(portfolio, path, *raw)["ecl"].sum(),
            }
        )
    table = pd.DataFrame(rows)
    table["allowance_ratio"] = table["allowance"] / balance
    base = table.loc[table["severity"] == 0, "allowance"].iloc[0]
    table["multiple_of_baseline"] = table["allowance"] / base

    # Severely adverse's own move relative to baseline, for scale.
    excess = changes["severely_adverse"] - baseline
    severe_unemployment = excess["unemployment_rate"].max()
    severe_house_prices = np.expm1(excess["hpi_log_change"].min())

    m = table["multiple_of_baseline"].to_numpy()
    above = np.nonzero(m >= threshold_multiple)[0]
    if len(above) and above[0] > 0:
        i = above[0]
        s0, s1 = table["severity"].iloc[i - 1], table["severity"].iloc[i]
        breaking = s0 + (threshold_multiple - m[i - 1]) / (m[i] - m[i - 1]) * (s1 - s0)
    else:
        breaking = np.nan
    implied = shock_at(breaking) if np.isfinite(breaking) else {}
    summary = pd.DataFrame(
        {
            "threshold_multiple": [threshold_multiple],
            "baseline_allowance": [base],
            "breaking_severity": [breaking],
            "implied_unemployment_rise": [implied.get("unemployment_rate", np.nan)],
            "implied_house_price_change": [implied.get("house_price_index", np.nan)],
            "severely_adverse_unemployment_rise": [severe_unemployment],
            "severely_adverse_house_price_change": [severe_house_prices],
        }
    )

    fig = go.Figure(
        go.Scatter(
            x=table["severity"], y=table["multiple_of_baseline"], mode="lines+markers",
            line_color="#083E44", name="allowance",
        )
    )
    fig.add_hline(
        y=threshold_multiple, line_dash="dash", line_color="#DE257E",
        annotation_text=f"{threshold_multiple:g}x baseline",
    )
    fig.update_layout(
        title=(
            "Allowance as a multiple of baseline vs shock severity "
            f"(1 = unemployment +{full_shock['unemployment_rate']:g}pp, "
            f"house prices {full_shock['house_price_index']:+.0%})"
        ),
        xaxis_title="severity",
        yaxis_title="multiple of baseline allowance",
        height=400,
    )

    return (
        {
            "Breaking point": summary.round(4),
            "Allowance by shock severity": table.round(6),
        },
        fig,
    )
