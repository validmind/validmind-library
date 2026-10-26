# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMModel

from .statsutils import INTERCEPT_NAMES, ols_results

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from statsmodels.stats.diagnostic import linear_reset
except ImportError as e:
    if "statsmodels" in str(e):
        raise MissingDependencyError(
            "Missing required package `statsmodels` for LinearitySpecification. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["statsmodels"],
            extra="stats",
        ) from e

    raise e


@tags("model_diagnosis", "specification", "visualization", "statsmodels")
@tasks("regression")
def LinearitySpecification(
    model: VMModel,
    reset_power: int = 3,
    significance_level: float = 0.05,
    max_points: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, go.Figure, bool, RawData]:
    """
    Tests whether a linear regression model's functional form is adequate, using Ramsey's
    RESET test and partial-residual plots for each continuous regressor.

    ### Purpose

    A linear model assumes each regressor affects the target along a straight line. When the
    true relationship curves — a diminishing effect, a threshold, an interaction — the
    coefficients describe an average slope that is wrong across much of the range, and
    predictions are biased in a systematic, feature-dependent way. This test checks the
    functional form as a whole and shows which regressor is responsible.

    ### Test Mechanism

    **RESET:** the model is re-estimated with powers of its fitted values (up to
    `reset_power`) added as regressors, and an F-test checks whether they add explanatory
    power. A p-value below `significance_level` rejects the linear specification.

    **Partial-residual (component-plus-residual) plots:** for every continuous regressor j,
    `residual + coefficient_j * x_j` is plotted against `x_j` with a LOWESS smoother. If the
    specification is right for that regressor, the smoother follows a straight line; a curve
    shows the shape the model is missing. Binary and dummy columns are skipped. At most
    `max_points` points per plot are drawn (sampled with `seed`); the smoother uses all data.

    The test passes when RESET does not reject.

    ### Signs of High Risk

    - RESET p-value below the significance level.
    - A partial-residual smoother that bends, flattens or changes direction across a
    regressor's range.

    ### Strengths

    - Combines a formal test of the whole specification with a per-regressor diagnosis.
    - Partial-residual plots suggest the fix (for example a log or a spline term).

    ### Limitations

    - Requires a statsmodels OLS model.
    - RESET also rejects for reasons other than a missing curve in one regressor — for example
    a target bounded at zero — so read it together with the plots.
    - In large samples RESET rejects small, immaterial departures from linearity.
    """
    results = ols_results(model, "LinearitySpecification")

    reset = linear_reset(results, power=reset_power, use_f=True)
    p_value = float(reset.pvalue)
    passed = bool(p_value >= significance_level)

    reset_df = pd.DataFrame(
        [
            {
                "Test": f"Ramsey RESET (powers 2..{reset_power} of fitted values)",
                "F Statistic": round(float(reset.fvalue), 4),
                "p-value": round(p_value, 6),
                "Significance Level": significance_level,
                "Pass/Fail": "Pass" if passed else "Fail",
            }
        ]
    )

    exog = np.asarray(results.model.exog, dtype=float)
    names = list(results.model.exog_names)
    resid = np.asarray(results.resid, dtype=float)
    params = np.asarray(results.params, dtype=float)

    continuous = [
        i
        for i, name in enumerate(names)
        if name not in INTERCEPT_NAMES and len(np.unique(exog[:, i])) > 2
    ]

    rng = np.random.default_rng(seed)
    shown = (
        rng.choice(len(resid), max_points, replace=False)
        if len(resid) > max_points
        else np.arange(len(resid))
    )

    n_cols = min(3, max(1, len(continuous)))
    n_rows = max(1, math.ceil(len(continuous) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[names[i] for i in continuous],
    )
    smoothers = {}
    for k, i in enumerate(continuous):
        x = exog[:, i]
        partial_residual = resid + params[i] * x
        # many tied values (a count with few levels) leave some local windows with no
        # spread in x; lowess divides by zero there and still returns a finite smooth
        with np.errstate(divide="ignore", invalid="ignore"):
            smooth = lowess(partial_residual, x, frac=0.3, return_sorted=True)
        smoothers[names[i]] = smooth
        row, col = k // n_cols + 1, k % n_cols + 1
        fig.add_trace(
            go.Scattergl(
                x=x[shown],
                y=partial_residual[shown],
                mode="markers",
                marker=dict(size=3, opacity=0.3, color="#5c6bc0"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x[[x.argmin(), x.argmax()]],
                y=params[i] * x[[x.argmin(), x.argmax()]] + resid.mean(),
                mode="lines",
                line=dict(color="#757575", dash="dash"),
                name="Linear fit",
                showlegend=k == 0,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=smooth[:, 0],
                y=smooth[:, 1],
                mode="lines",
                line=dict(color="#d81b60", width=2),
                name="LOWESS",
                showlegend=k == 0,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        title=f"Partial-Residual Plots — {model.input_id}",
        height=320 * n_rows,
    )

    return (
        reset_df,
        fig,
        passed,
        RawData(
            reset_p_value=p_value,
            lowess=smoothers,
            model=model.input_id,
        ),
    )
