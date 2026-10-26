# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

# scipy's Shapiro-Wilk p-value is only accurate up to 5000 observations
_SHAPIRO_MAX_N = 5000


@tags("model_diagnosis", "statistical_test", "normality", "visualization")
@tasks("regression")
def ResidualNormality(
    dataset: VMDataset,
    model: VMModel,
    significance_level: float = 0.05,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
    """
    Tests whether a regression model's residuals are normally distributed, using the
    Jarque-Bera, Shapiro-Wilk and Anderson-Darling tests and a Q-Q plot.

    ### Purpose

    Ordinary least squares does not need normal errors to estimate coefficients, but its
    p-values, confidence intervals and prediction intervals do — especially in small samples
    and in the tails. Skewed or heavy-tailed residuals also signal that the model misses part
    of the target's structure, such as a floor at zero or a sub-population. This test checks
    the residuals, not the features: feature normality is not an assumption of regression.
    `stats.NormalityTests` tests the features; `ModelPredictionResiduals` plots residuals
    with a single Kolmogorov-Smirnov test. This test adds Jarque-Bera, Shapiro-Wilk and
    Anderson-Darling, skewness and kurtosis, a Q-Q plot, and a pass/fail result.

    ### Test Mechanism

    Residuals are `actual - predicted` on the dataset, using the predictions assigned to it
    for the model. Three tests of normality are applied:

    - **Jarque-Bera:** based on skewness and kurtosis; sensitive to asymmetry and heavy tails.
    - **Shapiro-Wilk:** a general test with good power; if there are more than 5,000
    residuals, a random sample of 5,000 is tested (seeded by `seed`).
    - **Anderson-Darling:** gives extra weight to the tails; compared with its critical value
    at the significance level closest to `significance_level`.

    The Q-Q plot compares standardized residual quantiles with normal quantiles; points off
    the diagonal show where the distribution departs (one tail, both tails, a step).

    The test passes when none of the three tests rejects normality.

    ### Signs of High Risk

    - Rejection by all three tests, with large skewness or excess kurtosis.
    - A Q-Q plot with a long tail or a flat section (for example many residuals at one value
    from a target bounded at zero).

    ### Strengths

    - Assesses the actual model errors, for any regression model with predictions.
    - Combines tests with different sensitivities with a visual diagnosis.

    ### Limitations

    - In large samples the tests reject small, immaterial departures; use the Q-Q plot and
    the skewness and kurtosis to judge materiality.
    - Residuals on training data understate the errors on new data.
    """
    actual = np.asarray(dataset.y, dtype=float)
    residuals = actual - np.asarray(dataset.y_pred(model), dtype=float)
    n = len(residuals)

    jb_stat, jb_p = stats.jarque_bera(residuals)

    rng = np.random.default_rng(seed)
    shapiro_sample = (
        rng.choice(residuals, _SHAPIRO_MAX_N, replace=False)
        if n > _SHAPIRO_MAX_N
        else residuals
    )
    sw_stat, sw_p = stats.shapiro(shapiro_sample)

    anderson = stats.anderson(residuals, dist="norm")
    levels = np.asarray(anderson.significance_level) / 100
    level_idx = int(np.argmin(np.abs(levels - significance_level)))
    ad_critical = float(anderson.critical_values[level_idx])
    ad_level = float(levels[level_idx])

    tests_df = pd.DataFrame(
        [
            {
                "Test": "Jarque-Bera",
                "Statistic": float(jb_stat),
                "p-value": float(jb_p),
                "Fails When": f"p-value < {significance_level}",
                "Pass/Fail": "Pass" if jb_p >= significance_level else "Fail",
            },
            {
                "Test": f"Shapiro-Wilk (n={len(shapiro_sample)})",
                "Statistic": float(sw_stat),
                "p-value": float(sw_p),
                "Fails When": f"p-value < {significance_level}",
                "Pass/Fail": "Pass" if sw_p >= significance_level else "Fail",
            },
            {
                "Test": "Anderson-Darling",
                "Statistic": float(anderson.statistic),
                "p-value": np.nan,
                "Fails When": f"statistic > {ad_critical:.4f} (critical value at {ad_level:.1%})",
                "Pass/Fail": "Pass" if anderson.statistic <= ad_critical else "Fail",
            },
        ]
    )

    shape_df = pd.DataFrame(
        [
            {
                "Observations": n,
                "Mean Residual": float(residuals.mean()),
                "Std. Deviation": float(residuals.std(ddof=1)),
                "Skewness": float(stats.skew(residuals)),
                "Excess Kurtosis": float(stats.kurtosis(residuals)),
            }
        ]
    )

    standardized = np.sort((residuals - residuals.mean()) / residuals.std(ddof=1))
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Normal Q-Q Plot", "Residual Distribution"]
    )
    fig.add_trace(
        go.Scattergl(
            x=theoretical,
            y=standardized,
            mode="markers",
            marker=dict(size=3, color="#5c6bc0"),
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    bound = float(max(abs(theoretical).max(), 1))
    fig.add_trace(
        go.Scatter(
            x=[-bound, bound],
            y=[-bound, bound],
            mode="lines",
            line=dict(color="#757575", dash="dash"),
            name="Normal",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=60, marker_color="#5c6bc0", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
    fig.update_yaxes(title_text="Standardized Residual Quantiles", row=1, col=1)
    fig.update_xaxes(title_text="Residual", row=1, col=2)
    fig.update_layout(
        title=f"Residual Normality — {model.input_id} on {dataset.input_id}"
    )

    passed = bool((tests_df["Pass/Fail"] == "Pass").all())

    return (
        {"Normality Tests": tests_df.round(6), "Residual Shape": shape_df.round(4)},
        fig,
        passed,
        RawData(residuals=residuals, model=model.input_id, dataset=dataset.input_id),
    )
