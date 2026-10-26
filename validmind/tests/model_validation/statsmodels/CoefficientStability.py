# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMModel

from .statsutils import INTERCEPT_NAMES, ols_results


@tags("model_diagnosis", "robustness", "statsmodels")
@tasks("regression")
def CoefficientStability(
    model: VMModel,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    sign_stability_threshold: float = 0.95,
    seed: int = 42,
) -> Tuple[pd.DataFrame, bool, RawData]:
    """
    Assesses how stable a linear regression model's coefficients are under resampling of
    the fitting data, using bootstrap confidence intervals and sign stability.

    ### Purpose

    A coefficient is only useful if a similar sample would give a similar value. Analytical
    confidence intervals rely on the model's assumptions (normal, constant-variance errors);
    the bootstrap does not. Sign stability answers the question a business reader cares
    about: would another sample even agree on the direction of the effect?

    ### Test Mechanism

    The rows of the fitting data are resampled with replacement `n_bootstrap` times (seeded
    by `seed`), and the model is re-estimated on each resample. For each coefficient the test
    reports the original estimate, the analytical confidence interval, the bootstrap
    percentile interval at `confidence_level`, the ratio of their widths, and the share of
    resamples in which the coefficient keeps the sign of the original estimate. Resamples
    that cannot estimate every coefficient (a rare dummy with no rows) are skipped.

    The test passes when every coefficient except the intercept keeps its sign in at least
    `sign_stability_threshold` of resamples.

    ### Signs of High Risk

    - Sign stability below the threshold: the direction of the effect is not established.
    - Bootstrap intervals much wider than analytical ones, suggesting the model's assumptions
    understate the uncertainty.

    ### Strengths

    - Assumption-light: does not rely on normal or constant-variance errors.
    - Reports uncertainty in terms a business reader can act on (direction of effect).

    ### Limitations

    - Requires a statsmodels OLS model.
    - Resampling rows assumes observations are independent; it understates uncertainty for
    clustered or time-dependent data.
    - Coefficients of collinear regressors are legitimately unstable; read together with the
    VIF test.
    """
    results = ols_results(model, "CoefficientStability")
    exog = np.asarray(results.model.exog, dtype=float)
    endog = np.asarray(results.model.endog, dtype=float)
    names = list(results.model.exog_names)
    estimate = np.asarray(results.params, dtype=float)

    rng = np.random.default_rng(seed)
    n = len(endog)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        # a resample missing every row of a rare dummy cannot estimate it; lstsq would
        # report 0, which reads as a sign flip
        if np.linalg.matrix_rank(exog[idx]) == len(names):
            draws.append(np.linalg.lstsq(exog[idx], endog[idx], rcond=None)[0])
    if len(draws) < n_bootstrap / 2:
        raise SkipTestError(
            f"CoefficientStability: only {len(draws)} of {n_bootstrap} resamples could "
            "estimate every coefficient; a regressor is too rare to bootstrap"
        )
    draws = np.array(draws)

    alpha = 1 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    analytical = np.asarray(results.conf_int(alpha=alpha), dtype=float)
    sign_stability = (np.sign(draws) == np.sign(estimate)).mean(axis=0)

    pct = f"{confidence_level:.0%}"
    table = pd.DataFrame(
        {
            "Coefficient": names,
            "Estimate": estimate,
            f"Analytical {pct} CI Lower": analytical[:, 0],
            f"Analytical {pct} CI Upper": analytical[:, 1],
            f"Bootstrap {pct} CI Lower": lower,
            f"Bootstrap {pct} CI Upper": upper,
            "CI Width Ratio (Bootstrap / Analytical)": (upper - lower)
            / (analytical[:, 1] - analytical[:, 0]),
            "Sign Stability": sign_stability,
        }
    )
    is_intercept = table["Coefficient"].isin(INTERCEPT_NAMES)
    table["Pass/Fail"] = np.where(
        is_intercept,
        "N/A",
        np.where(sign_stability >= sign_stability_threshold, "Pass", "Fail"),
    )

    passed = bool((table.loc[~is_intercept, "Pass/Fail"] == "Pass").all())

    return (
        table.round(6),
        passed,
        RawData(
            bootstrap_coefficients=pd.DataFrame(draws, columns=names),
            rank_deficient_draws=n_bootstrap - len(draws),
            model=model.input_id,
        ),
    )
