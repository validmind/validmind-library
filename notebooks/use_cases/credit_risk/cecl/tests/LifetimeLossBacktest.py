# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cecl_model import actual_path, lifetime_ecl, portfolio_at
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("cecl", "credit_risk", "backtesting", "lifetime_loss")
@tasks("regression")
def LifetimeLossBacktest(
    datasets: list[VMDataset],
    models: list[VMModel],
    as_of: str = "2019Q4",
    tolerance: float = 0.25,
):
    """
    Re-runs the full lifetime loss engine at a past date on the economy that actually happened, and
    compares it with the losses that followed.

    ### Purpose

    Component tests check the default, prepayment and LGD models one at a time. The allowance
    depends on how they combine: survival compounds the default and prepayment hazards over many
    quarters, and LGD is applied at projected collateral values. This test checks the whole chain
    the way an auditor would. It books the allowance at a past date, feeds the engine the
    macroeconomic path that actually followed, and asks whether the defaults, prepayments and
    losses it predicted for that portfolio happened.

    Using the **actual** macro path takes the scenario forecast out of the comparison, so any gap
    is model error rather than forecast error.

    ### Test Mechanism

    `datasets` is the loan-quarter history (development and out-of-time panels, concatenated) and
    `models` are the default, prepayment and LGD models, in that order. The test takes the loans
    on the books at the end of `as_of`, projects them with the realized unemployment, mortgage
    rate and house-price path, and compares quarter by quarter, over the quarters with actual
    data, the expected number of defaults and prepayments and the expected loss with what those
    same loans realized. The test passes when cumulative realized loss is within `tolerance`
    (relative) of cumulative expected loss.

    Choose `as_of` at or after the end of the development window so the comparison is out of
    time.

    ### Signs of High Risk

    - Cumulative expected loss far from realized loss: the allowance would have been materially
      wrong
    - Expected defaults close to realized but losses far off, which points to LGD or exposure
    - Expected prepayments far off, which distorts every loan's remaining life and so the lifetime
      loss even when quarterly default rates are right

    ### Strengths

    - Tests the allowance engine end to end, including survival compounding, amortization and
      the interaction between models
    - Isolates model error from scenario error by using realized macro data

    ### Limitations

    - Covers only the quarters since `as_of`, not the full remaining life
    - One portfolio and one economic path: a single historical episode can flatter or penalize
      the model by chance
    """
    panel = pd.concat([ds.df for ds in datasets], ignore_index=True)
    panel["quarter"] = panel["quarter"].astype(str)
    default_model, prepay_model, lgd_model = (m.model for m in models)

    portfolio = portfolio_at(panel, as_of)
    path = actual_path(as_of)
    n_actual = path.attrs["actual_quarters"]
    result = lifetime_ecl(portfolio, path, default_model, prepay_model, lgd_model)
    quarters = path.index[:n_actual].astype(str)

    expected = pd.DataFrame(
        {
            "expected_defaults": result["marginal_pd"][:, :n_actual].sum(axis=0),
            "expected_prepayments": result["marginal_pp"][:, :n_actual].sum(axis=0),
            "expected_loss": result["loss"][:, :n_actual].sum(axis=0),
        },
        index=quarters[: result["loss"].shape[1]],
    )
    later = panel[
        panel["loan_id"].isin(portfolio["loan_id"]) & panel["quarter"].isin(quarters)
    ]
    g = later.groupby("quarter")
    realized = pd.DataFrame(
        {
            "actual_defaults": g["outcome"].apply(lambda s: (s == 1).sum()),
            "actual_prepayments": g["outcome"].apply(lambda s: (s == 2).sum()),
            "actual_loss": g["loss"].sum(),
        }
    )
    by_quarter = expected.join(realized, how="left").fillna(0)

    by_year = by_quarter.groupby(by_quarter.index.str[:4]).sum()
    by_year.loc["total"] = by_year.sum()
    for k in ["defaults", "prepayments", "loss"]:
        by_year[f"{k}_actual_to_expected"] = by_year[f"actual_{k}"] / by_year[
            f"expected_{k}"
        ].replace(0, np.nan)

    total = by_year.loc["total"]
    loss_ratio = total["actual_loss"] / total["expected_loss"]
    summary = pd.DataFrame(
        {
            "as_of": [as_of],
            "loans": [len(portfolio)],
            "balance": [portfolio["balance"].sum()],
            "quarters_compared": [len(by_quarter)],
            "expected_loss": [total["expected_loss"]],
            "actual_loss": [total["actual_loss"]],
            "loss_actual_to_expected": [loss_ratio],
            "tolerance": [tolerance],
        }
    )

    x = pd.PeriodIndex(by_quarter.index, freq="Q").to_timestamp()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Cumulative loss ($m)",
            "Cumulative defaults",
            "Cumulative prepayments",
        ],
    )
    for i, (k, scale) in enumerate(
        [("loss", 1e6), ("defaults", 1), ("prepayments", 1)], start=1
    ):
        for kind, color in [("expected", "#DE257E"), ("actual", "#083E44")]:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=by_quarter[f"{kind}_{k}"].cumsum() / scale,
                    name=kind,
                    line_color=color,
                    showlegend=i == 1,
                ),
                row=1,
                col=i,
            )
    fig.update_layout(
        title=f"Portfolio as of {as_of}, projected on the realized macro path",
        height=380,
    )

    return (
        {
            "Summary": summary.round(4),
            "Back-test by year": by_year.reset_index(names="year").round(4),
        },
        fig,
        bool(abs(loss_ratio - 1) <= tolerance),
    )
