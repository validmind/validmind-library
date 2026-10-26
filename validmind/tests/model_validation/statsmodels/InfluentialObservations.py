# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMModel

from .statsutils import ols_results, row_labels


def _externally_studentized(results, leverage):
    """
    Externally studentized residuals in closed form.

    Equal to statsmodels' `resid_studentized_external`, which refits the model once per
    observation and takes minutes on a few thousand rows.
    """
    resid = np.asarray(results.resid, dtype=float)
    dof = results.df_resid
    sse = resid**2 / (1 - leverage)
    sigma_without_i = np.sqrt((dof * results.scale - sse) / (dof - 1))
    return resid / (sigma_without_i * np.sqrt(1 - leverage))


@tags("model_diagnosis", "outliers", "visualization", "statsmodels")
@tasks("regression")
def InfluentialObservations(
    model: VMModel,
    cooks_threshold: Optional[float] = None,
    leverage_threshold: Optional[float] = None,
    studentized_threshold: float = 3.0,
    severe_cooks_threshold: float = 1.0,
    top_n: int = 10,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
    """
    Identifies observations that individually drive a linear regression model's estimates,
    using Cook's distance, leverage and studentized residuals, and shows how much the
    coefficients move when the most influential observations are removed.

    ### Purpose

    Ordinary least squares weights every observation by its squared error, so a handful of
    unusual rows — data entry errors, a different population, genuine extremes — can pull the
    coefficients far from what the rest of the data supports. A validator needs to know
    whether the model is a description of the portfolio or of a few rows.

    ### Test Mechanism

    For every observation in the fitting data the test computes:

    - **Leverage** (hat value): how unusual the observation's regressors are. Flagged above
    `leverage_threshold` (default `2p/n`, p = number of coefficients).
    - **Studentized residual**: how unusual its target is given the model. Flagged above
    `studentized_threshold` in absolute value (default 3).
    - **Cook's distance**: the combined effect of leverage and residual on all fitted values.
    Flagged above `cooks_threshold` (default `4/n`); values above `severe_cooks_threshold`
    (default 1) mean a single observation moves the fit materially.

    The model is then re-fitted without the severe observations, and each coefficient's
    change is reported. The influence plot shows leverage against studentized residuals,
    with marker size proportional to Cook's distance.

    The test passes when no observation exceeds `severe_cooks_threshold`.

    ### Signs of High Risk

    - Any observation with Cook's distance above 1.
    - Coefficients that change sign or materially in size when a few observations are removed.
    - Influential observations with implausible values (often data errors).

    ### Strengths

    - Pinpoints the specific rows to investigate, by their index in the fitting data.
    - Quantifies their impact on each coefficient, not just on fit.

    ### Limitations

    - Requires a statsmodels OLS model.
    - The `4/n` and `2p/n` thresholds flag many observations in large samples; they are
    reported as counts for context, and only the severe threshold decides pass/fail.
    - Influence measures assess one observation at a time and can miss a cluster of
    observations that mask each other.
    """
    results = ols_results(model, "InfluentialObservations")
    n = int(results.nobs)
    p = len(results.params)
    cooks_threshold = cooks_threshold if cooks_threshold is not None else 4 / n
    leverage_threshold = (
        leverage_threshold if leverage_threshold is not None else 2 * p / n
    )

    influence = results.get_influence()
    cooks = np.asarray(influence.cooks_distance[0])
    leverage = np.asarray(influence.hat_matrix_diag)
    studentized = _externally_studentized(results, leverage)

    measures = pd.DataFrame(
        {
            "Observation": row_labels(results),
            "Cook's Distance": cooks,
            "Leverage": leverage,
            "Studentized Residual": studentized,
        }
    )

    summary_df = pd.DataFrame(
        [
            {
                "Measure": "Cook's distance",
                "Threshold": round(cooks_threshold, 6),
                "Observations Above": int((cooks > cooks_threshold).sum()),
            },
            {
                "Measure": "Cook's distance (severe)",
                "Threshold": severe_cooks_threshold,
                "Observations Above": int((cooks > severe_cooks_threshold).sum()),
            },
            {
                "Measure": "Leverage",
                "Threshold": round(leverage_threshold, 6),
                "Observations Above": int((leverage > leverage_threshold).sum()),
            },
            {
                "Measure": "|Studentized residual|",
                "Threshold": studentized_threshold,
                "Observations Above": int(
                    (np.abs(studentized) > studentized_threshold).sum()
                ),
            },
        ]
    )

    top_df = (
        measures.sort_values("Cook's Distance", ascending=False)
        .head(top_n)
        .round(4)
        .reset_index(drop=True)
    )

    severe = cooks > severe_cooks_threshold
    if severe.any():
        keep = ~severe
        refit = type(results.model)(
            results.model.endog[keep], results.model.exog[keep]
        ).fit()
        impact_df = pd.DataFrame(
            {
                "Coefficient": results.model.exog_names,
                "All Observations": results.params.values,
                f"Without {int(severe.sum())} Severe": refit.params,
            }
        )
        impact_df["Change"] = (
            impact_df[f"Without {int(severe.sum())} Severe"]
            - impact_df["All Observations"]
        )
        impact_df = impact_df.round(4)
    else:
        impact_df = pd.DataFrame(
            [{"Result": "No observation exceeds the severe Cook's distance threshold"}]
        )

    fig = go.Figure(
        go.Scattergl(
            x=leverage,
            y=studentized,
            mode="markers",
            text=[f"Observation {o}" for o in measures["Observation"]],
            marker=dict(
                size=4 + 30 * np.sqrt(cooks / max(cooks.max(), 1e-12)),
                color=np.where(cooks > cooks_threshold, "#c62828", "#5c6bc0"),
                opacity=0.6,
            ),
        )
    )
    fig.add_vline(x=leverage_threshold, line_dash="dash", line_color="#757575")
    for y in (studentized_threshold, -studentized_threshold):
        fig.add_hline(y=y, line_dash="dash", line_color="#757575")
    fig.update_layout(
        title=f"Influence Plot — {model.input_id} (marker size ∝ Cook's distance)",
        xaxis_title="Leverage",
        yaxis_title="Studentized Residual",
    )

    return (
        {
            "Influence Summary": summary_df,
            f"Top {top_n} Observations by Cook's Distance": top_df,
            "Coefficient Impact of Severe Observations": impact_df,
        },
        fig,
        bool(not severe.any()),
        RawData(influence_measures=measures, model=model.input_id),
    )
