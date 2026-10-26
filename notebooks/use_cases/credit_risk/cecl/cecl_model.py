# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Lifetime expected credit loss engine for the CECL residential mortgage notebook.

Shared by the notebook and by the CECL tests in ``tests/``, so a test that re-runs the
allowance under different assumptions runs exactly the calculation the notebook reports.
"""

import numpy as np
import pandas as pd

from cecl_data import REPORTING_QUARTER, load_fed_scenarios, load_macro_history

FEATURES = [
    "fico",
    "current_ltv",
    "dti",
    "occupancy",
    "purpose",
    "term_months",
    "loan_age",
    "unemployment_rate",
    "refi_incentive",
]
CATEGORIES = {
    "occupancy": ["owner", "investor", "second_home"],
    "purpose": ["purchase", "rate_term_refi", "cash_out_refi"],
}

RS_QUARTERS = 8
REVERSION_QUARTERS = 4
MAX_HORIZON = 120  # 30 years
SCENARIO_WEIGHTS = {"baseline": 0.6, "adverse": 0.3, "severely_adverse": 0.1}
MACRO_COLUMNS = ["unemployment_rate", "mortgage_rate", "house_price_index"]


def model_features(df):
    """The model's feature frame, with categoricals typed for the gradient-boosted models."""
    X = df[FEATURES].copy()
    for col, categories in CATEGORIES.items():
        X[col] = pd.Categorical(X[col], categories=categories)
    return X


def scheduled_balance(orig_amount, note_rate, term_months, months_paid):
    """Contractual balance after ``months_paid`` payments. Broadcasts over arrays."""
    r = note_rate / 1200
    paid = np.minimum(months_paid, term_months)
    growth = (1 + r) ** term_months
    return orig_amount * (growth - (1 + r) ** paid) / (growth - 1)


def long_run_averages(through=REPORTING_QUARTER):
    history = load_macro_history().loc["1990Q1":through]
    return {
        "unemployment_rate": history["unemployment_rate"].mean(),
        "mortgage_rate": history["mortgage_rate"].mean(),
        "hpi_growth": np.log(history["house_price_index"]).diff().mean(),
    }


def _changes(levels, jump):
    """Changes from a jump-off quarter: unemployment and mortgage rate in points, HPI in logs."""
    return pd.DataFrame(
        {
            "unemployment_rate": levels["unemployment_rate"].to_numpy()
            - jump["unemployment_rate"],
            "mortgage_rate": levels["mortgage_rate"].to_numpy() - jump["mortgage_rate"],
            "hpi_log_change": np.log(
                levels["house_price_index"].to_numpy() / jump["house_price_index"]
            ),
        }
    )


def scenario_changes(rs_quarters=RS_QUARTERS):
    """Each Fed 2025 scenario's change from its own jump-off quarter, per forecast quarter.

    Adverse is the midpoint of baseline and severely adverse.
    """
    fed = load_fed_scenarios()
    jump = fed[fed["scenario"] == "jump_off"].iloc[0]
    changes = {}
    for s in ["baseline", "severely_adverse"]:
        f = fed[fed["scenario"] == s]
        if rs_quarters > len(f):
            raise ValueError(
                f"The Fed scenarios cover {len(f)} quarters, not {rs_quarters}"
            )
        changes[s] = _changes(f.iloc[:rs_quarters], jump)
    changes["adverse"] = (changes["baseline"] + changes["severely_adverse"]) / 2
    return changes


def historical_changes(start, rs_quarters=RS_QUARTERS):
    """The macro changes that actually followed ``start``, over ``rs_quarters`` quarters."""
    macro = load_macro_history()
    start = pd.Period(start, freq="Q")
    following = macro.loc[start + 1 : start + rs_quarters]
    if len(following) < rs_quarters:
        raise ValueError(f"History covers only {len(following)} quarters after {start}")
    return _changes(following, macro.loc[start])


def path_from_changes(
    changes,
    reversion_quarters=REVERSION_QUARTERS,
    reporting_quarter=REPORTING_QUARTER,
    horizon=MAX_HORIZON,
):
    """A quarterly macro path for ``horizon`` quarters after the reporting date.

    ``changes`` (one row per forecast quarter, from ``scenario_changes`` or
    ``historical_changes``) are applied to the reporting-quarter actuals. The forecast
    holds for ``len(changes)`` quarters, then unemployment, the mortgage rate and the
    house-price growth rate revert in a straight line to their long-run averages since
    1990 over ``reversion_quarters`` and stay there. House prices revert in growth, not
    level, so a fall during the forecast is never recovered by reversion alone.
    """
    macro = load_macro_history()
    reporting = pd.Period(reporting_quarter, freq="Q")
    long_run = long_run_averages(reporting_quarter)
    actual = macro.loc[reporting]
    rs_quarters = len(changes)
    h = np.arange(1, horizon + 1)
    # Weight on the forecast: 1 through the R&S period, falling linearly to 0 over
    # reversion. With no reversion period the path jumps straight to long run.
    w = np.clip(1 - (h - rs_quarters) / (reversion_quarters + 1), 0, 1)
    pad = horizon - rs_quarters

    def blend(forecast, long_run_value):
        held = np.r_[forecast, np.repeat(forecast[-1], pad)]
        return w * held + (1 - w) * long_run_value

    hpi_growth = np.diff(np.r_[0, changes["hpi_log_change"]])
    return pd.DataFrame(
        {
            "horizon": h,
            "unemployment_rate": blend(
                actual["unemployment_rate"] + changes["unemployment_rate"].to_numpy(),
                long_run["unemployment_rate"],
            ),
            "mortgage_rate": blend(
                actual["mortgage_rate"] + changes["mortgage_rate"].to_numpy(),
                long_run["mortgage_rate"],
            ),
            "house_price_index": actual["house_price_index"]
            * np.exp(np.cumsum(blend(hpi_growth, long_run["hpi_growth"]))),
        },
        index=pd.period_range(reporting + 1, periods=horizon, freq="Q"),
    )


def shocked_changes(
    changes, unemployment_rate=0.0, house_price_index=0.0, mortgage_rate=0.0
):
    """``changes`` plus a shock that builds in a straight line to its full size by the
    last forecast quarter: unemployment and mortgage rate in points, house prices as a
    fractional change in level (-0.2 is a 20% fall)."""
    ramp = np.arange(1, len(changes) + 1) / len(changes)
    out = changes.copy()
    out["unemployment_rate"] += unemployment_rate * ramp
    out["mortgage_rate"] += mortgage_rate * ramp
    out["hpi_log_change"] += np.log1p(house_price_index) * ramp
    return out


def build_scenario_paths(
    rs_quarters=RS_QUARTERS,
    reversion_quarters=REVERSION_QUARTERS,
    reporting_quarter=REPORTING_QUARTER,
    horizon=MAX_HORIZON,
):
    """Quarterly macro paths per scenario for ``horizon`` quarters after the reporting date.

    Each Fed 2025 scenario's change from its own jump-off quarter is applied to the
    reporting-quarter actuals; see ``path_from_changes``.
    """
    return {
        s: path_from_changes(changes, reversion_quarters, reporting_quarter, horizon)
        for s, changes in scenario_changes(rs_quarters).items()
    }


def actual_path(as_of, horizon=MAX_HORIZON):
    """The macro path that actually happened after ``as_of``, for back-testing.

    Actuals run to the last quarter of history; beyond it the path continues at the
    long-run averages, so a back-test should only be read over the actual quarters,
    recorded in ``path.attrs["actual_quarters"]``.
    """
    macro = load_macro_history()
    start = pd.Period(as_of, freq="Q") + 1
    realized = macro.loc[start:, MACRO_COLUMNS]
    n_actual = len(realized)
    long_run = long_run_averages()

    path = realized.reindex(pd.period_range(start, periods=horizon, freq="Q"))
    path["unemployment_rate"] = path["unemployment_rate"].fillna(
        long_run["unemployment_rate"]
    )
    path["mortgage_rate"] = path["mortgage_rate"].fillna(long_run["mortgage_rate"])
    steps = np.arange(1, horizon - n_actual + 1)
    path.iloc[n_actual:, path.columns.get_loc("house_price_index")] = realized[
        "house_price_index"
    ].iloc[-1] * np.exp(long_run["hpi_growth"] * steps)
    path.insert(0, "horizon", np.arange(1, horizon + 1))
    path.attrs["actual_quarters"] = n_actual
    return path


LOAN_COLUMNS = [
    "loan_id",
    "loan_age",
    "orig_quarter",
    "fico",
    "orig_ltv",
    "dti",
    "occupancy",
    "purpose",
    "term_months",
    "orig_amount",
    "note_rate",
    "property_value",
]


def portfolio_at(panel, as_of):
    """Loans on the books at the end of ``as_of``, rolled to the first forecast quarter."""
    macro = load_macro_history()
    quarter = panel["quarter"].astype(str)
    p = panel[(quarter == str(as_of)) & (panel["outcome"] == 0)].copy()
    p = p.drop(
        columns=[c for c in p.columns if c not in LOAN_COLUMNS], errors="ignore"
    ).reset_index(drop=True)
    p["loan_age"] += 1
    p["balance"] = scheduled_balance(
        p["orig_amount"], p["note_rate"], p["term_months"], 3 * (p["loan_age"] - 1)
    )
    orig = pd.PeriodIndex(p["orig_quarter"].astype(str), freq="Q")
    p["orig_hpi"] = macro.loc[orig, "house_price_index"].to_numpy()
    p["current_ltv"] = p["balance"] / (
        p["property_value"]
        * macro.loc[pd.Period(as_of, freq="Q"), "house_price_index"]
        / p["orig_hpi"]
    )
    return p


def project(portfolio, path):
    """Each loan's features in every forecast quarter, as one long frame (loan-major).

    Returns the frame plus (loans x quarters) arrays of scheduled balance and an
    on-book mask that ends at contractual maturity.
    """
    n = len(portfolio)
    remaining = (portfolio["term_months"] // 3 - portfolio["loan_age"] + 1).to_numpy()
    horizon = min(int(remaining.max()), len(path))
    h = np.arange(horizon)

    age = portfolio["loan_age"].to_numpy()[:, None] + h
    col = lambda name: portfolio[name].to_numpy()[:, None]  # noqa: E731
    balance = scheduled_balance(
        col("orig_amount"), col("note_rate"), col("term_months"), 3 * (age - 1)
    )
    hpi = path["house_price_index"].to_numpy()[:horizon]
    value = col("property_value") * hpi / col("orig_hpi")
    mortgage_rate = path["mortgage_rate"].to_numpy()[:horizon]
    unemployment = path["unemployment_rate"].to_numpy()[:horizon]

    frame = pd.DataFrame(
        {
            "fico": np.repeat(portfolio["fico"].to_numpy(), horizon),
            "current_ltv": (balance / value).ravel(),
            "dti": np.repeat(portfolio["dti"].to_numpy(), horizon),
            "occupancy": np.repeat(portfolio["occupancy"].to_numpy(), horizon),
            "purpose": np.repeat(portfolio["purpose"].to_numpy(), horizon),
            "term_months": np.repeat(portfolio["term_months"].to_numpy(), horizon),
            "loan_age": age.ravel(),
            "unemployment_rate": np.tile(unemployment, n),
            "refi_incentive": (col("note_rate") - mortgage_rate).ravel(),
        }
    )
    on_book = h[None, :] < remaining[:, None]
    return frame, balance, on_book


def lifetime_ecl(portfolio, path, default_model, prepay_model, lgd_model):
    """Lifetime expected credit loss per loan under one macro path.

    Every loan is projected quarter by quarter to contractual maturity. Default and
    prepayment hazards come from the two cause-specific models, LGD from the LGD model
    at the projected mark-to-market LTV, and exposure is the scheduled balance:

        ECL_i = sum_h S_{i,h-1} * PD_{i,h} * LGD_{i,h} * EAD_{i,h}
        S_{i,h} = S_{i,h-1} * (1 - PD_{i,h} - PP_{i,h})

    Returns a dict of (loans x quarters) arrays: ``loss``, ``marginal_pd`` and
    ``marginal_pp`` (the unconditional probability of defaulting / prepaying in each
    quarter), plus per-loan ``ecl`` and ``lifetime_pd``.
    """
    frame, balance, on_book = project(portfolio, path)
    shape = balance.shape
    X = model_features(frame)
    pd_q = default_model.predict_proba(X)[:, 1].reshape(shape) * on_book
    pp_q = prepay_model.predict_proba(X)[:, 1].reshape(shape) * on_book
    lgd = lgd_model.predict(X).clip(0, 1).reshape(shape)

    survival = np.cumprod(1 - pd_q - pp_q, axis=1)
    survival_before = np.hstack([np.ones((shape[0], 1)), survival[:, :-1]])
    marginal_pd = survival_before * pd_q
    loss = marginal_pd * lgd * balance
    return {
        "loss": loss,
        "marginal_pd": marginal_pd,
        "marginal_pp": survival_before * pp_q,
        "ecl": loss.sum(axis=1),
        "lifetime_pd": marginal_pd.sum(axis=1),
    }


def weighted_allowance(portfolio, paths, models, weights=None):
    """Probability-weighted allowance and the per-scenario results behind it."""
    weights = weights or SCENARIO_WEIGHTS
    results = {s: lifetime_ecl(portfolio, paths[s], *models) for s in weights}
    total = sum(w * results[s]["ecl"].sum() for s, w in weights.items())
    return total, results


SEGMENT_BANDS = {
    "fico_band": ("fico", [0, 660, 700, 740, 780, 900]),
    "current_ltv_band": ("current_ltv", [0, 0.6, 0.8, 0.9, 1.0, 5]),
}


def segments(df):
    """Segment keys used to break the allowance down: vintage, FICO and LTV bands, occupancy."""
    keys = {"vintage": df["orig_quarter"].astype(str).str[:4].astype(int)}
    for name, (col, edges) in SEGMENT_BANDS.items():
        keys[name] = pd.cut(df[col], edges).astype(str)
    keys["occupancy"] = df["occupancy"]
    return keys
