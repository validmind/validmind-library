# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Data for the CECL residential mortgage notebook.

Macroeconomic history is real: quarterly averages of FRED series UNRATE (unemployment
rate), CSUSHPINSA (S&P CoreLogic Case-Shiller U.S. National Home Price Index) and
MORTGAGE30US (Freddie Mac 30-year fixed rate), 1990Q1-2026Q3. Forward-looking scenarios
are the Federal Reserve's 2025 supervisory baseline and severely adverse scenarios.

The loan portfolio is synthetic. Each loan is originated with borrower and collateral
attributes typical of its vintage, then walked quarter by quarter through the real macro
history. Every quarter it either defaults, prepays or stays current, drawn from a
competing-risk hazard that responds to credit quality, mark-to-market LTV, unemployment
and refinance incentive. Two features of the real 2004-2026 experience are built in so the
data does not flatter a model fitted to it:

- 2004-2007 originations carry an unobserved "low documentation" layer that raises
  default risk but is not reported in the loan tape, as was the case for stated-income
  loans.
- 2020Q2-2021Q4 defaults are dampened by forbearance: the unemployment shock feeds the
  hazard only partially, as it did under the CARES Act.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
REPORTING_QUARTER = "2026Q3"


def load_macro_history():
    """Quarterly macro history, indexed by quarter (PeriodIndex)."""
    df = pd.read_csv(DATA_DIR / "macro_history.csv")
    return df.set_index(pd.PeriodIndex(df.pop("quarter"), freq="Q"))


def load_fed_scenarios():
    """Fed 2025 supervisory scenarios plus their 2024Q4 jump-off quarter."""
    df = pd.read_csv(DATA_DIR / "fed_2025_supervisory_scenarios.csv")
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df


def _amortized_balance(amount, monthly_rate, term_months, months_paid):
    growth = (1 + monthly_rate) ** term_months
    paid = (1 + monthly_rate) ** np.minimum(months_paid, term_months)
    return amount * (growth - paid) / (growth - 1)


def _originate(rng, quarter, macro, start_id):
    rate = macro.loc[quarter, "mortgage_rate"]
    hpi = macro.loc[quarter, "house_price_index"]
    year = quarter.year
    boom = 2004 <= year <= 2007

    n = int(rng.poisson(260 * np.exp(-0.18 * (rate - 5.5))))
    fico = np.clip(rng.normal(712 if boom else 748, 55 if boom else 42, n), 580, 840)
    high_ltv = rng.random(n) < (0.28 if boom else 0.14)
    ltv = np.where(
        high_ltv, rng.uniform(90, 97, n), np.clip(rng.normal(74, 11, n), 30, 90)
    )
    dti = np.clip(rng.normal(38 if boom else 35, 9 if boom else 7, n), 10, 55)
    occupancy = rng.choice(
        ["owner", "investor", "second_home"], n, p=[0.87, 0.09, 0.04]
    )
    refi_share = np.clip(0.75 - 0.1 * (rate - 4), 0.2, 0.75)
    purpose = rng.choice(
        ["purchase", "rate_term_refi", "cash_out_refi"],
        n,
        p=[1 - refi_share, refi_share * 0.6, refi_share * 0.4],
    )
    term = np.where(rng.random(n) < 0.15, 180, 360)
    amount = np.round(
        np.exp(rng.normal(np.log(260_000), 0.45, n)) * hpi / 200, -3
    )  # loan size tracks house prices
    note_rate = (
        rate
        + 0.2
        - 0.35 * (term == 180)
        + 0.004 * (740 - fico).clip(0)
        + 0.25 * (ltv > 90)
        + 0.4 * (occupancy == "investor")
        + rng.normal(0, 0.12, n)
    )
    return pd.DataFrame(
        {
            "loan_id": np.arange(start_id, start_id + n),
            "orig_quarter": quarter,
            "fico": fico.round(),
            "orig_ltv": ltv.round(1),
            "dti": dti.round(1),
            "occupancy": occupancy,
            "purpose": purpose,
            "term_months": term,
            "orig_amount": amount,
            "note_rate": note_rate.round(3),
            "property_value": np.round(amount / (ltv / 100), -2),
            "orig_hpi": hpi,
            # Unobserved in the loan tape: stated-income / low-doc layering.
            "_low_doc": boom & (rng.random(n) < 0.3),
        }
    )


def simulate_portfolio(seed=42, first_quarter="2004Q1"):
    """Simulate originations and quarterly performance through the reporting quarter.

    Returns:
        loans: one row per loan with origination attributes and final status.
        panel: one row per loan-quarter while the loan was active, with the
            quarter's outcome: 0 = current, 1 = default, 2 = prepay. ``balance`` is the
            balance at the start of the quarter; ``loss`` is the realized loss on
            default rows.
    """
    rng = np.random.default_rng(seed)
    macro = load_macro_history()
    quarters = pd.period_range(first_quarter, REPORTING_QUARTER, freq="Q")
    unemp_change = macro["unemployment_rate"].diff(4)

    loans, rows, active = [], [], None
    for q in quarters:
        # Performance this quarter for loans already on the books.
        if active is not None and len(active):
            a = active
            age = (q - a["orig_quarter"]).apply(lambda d: d.n).to_numpy()
            r = a["note_rate"].to_numpy() / 1200
            balance = _amortized_balance(
                a["orig_amount"].to_numpy(),
                r,
                a["term_months"].to_numpy(),
                3 * (age - 1),
            )
            hpi_ratio = macro.loc[q, "house_price_index"] / a["orig_hpi"].to_numpy()
            cltv = balance / (a["property_value"].to_numpy() * hpi_ratio)
            unemp = macro.loc[q, "unemployment_rate"]
            mkt_rate = macro.loc[q, "mortgage_rate"]
            incentive = a["note_rate"].to_numpy() - mkt_rate
            in_forbearance = pd.Period("2020Q2") <= q <= pd.Period("2021Q4")
            unemp_effective = 0.25 * unemp + 0.75 * 3.8 if in_forbearance else unemp

            fico = a["fico"].to_numpy()
            occ = a["occupancy"].to_numpy()
            z_default = (
                -7.65
                - 0.95 * (fico - 720) / 50
                + 2.6 * np.clip(cltv - 0.8, -0.4, 0.9)
                + 0.025 * (a["dti"].to_numpy() - 36)
                + 0.32 * (unemp_effective - 5)
                + 0.12 * max(unemp_change.loc[q], 0) * (not in_forbearance)
                + 0.55 * (occ == "investor")
                + 0.25 * (a["purpose"].to_numpy() == "cash_out_refi")
                - 0.45 * (a["term_months"].to_numpy() == 180)
                + 0.75 * a["_low_doc"].to_numpy()
                + 0.55 * np.log(age)
                - 0.035 * age
                + rng.normal(0, 0.08)  # common quarterly frailty
            )
            z_prepay = (
                -4.75
                + 1.2 * np.clip(incentive, 0, 2.5)  # refinance into a lower rate
                + 0.35 * np.clip(incentive, -2.5, 0)  # lock-in dampens home sales
                + 1.2 * np.clip(0.8 - cltv, -0.4, 0.5)
                + 0.004 * (fico - 720)
                + 0.9 * np.minimum(age, 10) / 10
                + 0.35 * (a["term_months"].to_numpy() == 180)
                - 0.3 * (occ == "investor")
                + rng.normal(0, 0.06)
            )
            ez_d, ez_p = np.exp(z_default), np.exp(z_prepay)
            p_default = ez_d / (1 + ez_d + ez_p)
            p_prepay = ez_p / (1 + ez_d + ez_p)
            u = rng.random(len(a))
            outcome = np.where(
                u < p_default, 1, np.where(u < p_default + p_prepay, 2, 0)
            )
            matured = 3 * age >= a["term_months"].to_numpy()
            outcome = np.where(matured & (outcome == 0), 2, outcome)

            # Severity rises with negative equity; investor properties sell at a
            # deeper discount. Losses are recognised in the default quarter.
            lgd = np.clip(
                0.17
                + 0.75 * np.clip(cltv - 0.6, 0, None)
                + 0.08 * (occ == "investor")
                + rng.normal(0, 0.11, len(a)),
                0,
                1,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "loan_id": a["loan_id"].to_numpy(),
                        "quarter": q,
                        "loan_age": age,
                        "balance": balance.round(2),
                        "current_ltv": cltv.round(4),
                        "unemployment_rate": unemp,
                        "mortgage_rate": mkt_rate,
                        "refi_incentive": incentive.round(3),
                        "outcome": outcome,
                        "lgd": np.where(outcome == 1, lgd.round(4), np.nan),
                        "loss": np.where(outcome == 1, (lgd * balance).round(2), 0.0),
                    }
                )
            )
            active = a[outcome == 0]

        if q < pd.Period(REPORTING_QUARTER):
            new = _originate(rng, q, macro, sum(len(x) for x in loans))
            loans.append(new)
            active = new if active is None else pd.concat([active, new])

    loans = pd.concat(loans, ignore_index=True)
    panel = pd.concat(rows, ignore_index=True)
    status = panel.groupby("loan_id")["outcome"].last()
    loans["status"] = (
        loans["loan_id"]
        .map(status)
        .map({0: "active", 1: "defaulted", 2: "paid_off"})
        .fillna("active")
    )
    loans = loans.drop(columns=["orig_hpi", "_low_doc"])
    panel = panel.merge(loans.drop(columns="status"), on="loan_id", how="left")
    return loans, panel
