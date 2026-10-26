# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel

from .statsutils import design_data, ols_results


@tags("model_performance", "uncertainty", "visualization", "statsmodels")
@tasks("regression")
def PredictionIntervals(
    dataset: VMDataset,
    model: VMModel,
    confidence_level: float = 0.95,
    coverage_tolerance: float = 0.02,
    n_bins: int = 10,
    max_points: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, go.Figure, bool, RawData]:
    """
    Evaluates whether a linear regression model's prediction intervals are honest, by
    measuring how many actual values fall inside them on a dataset.

    ### Purpose

    A prediction is only as useful as the statement of its uncertainty. A 95% prediction
    interval claims that 95% of new observations will fall inside it; if only 80% do, every
    decision that relies on the interval — a limit, a reserve, a stress buffer — is
    under-protected. Backtesting coverage on held-out or out-of-time data is the direct test
    of that claim, and coverage by prediction range shows where it breaks.

    ### Test Mechanism

    For every row of the dataset, the model's prediction interval at `confidence_level` is
    computed from the fitted model (for OLS: the prediction ± the t-quantile times the
    standard error of a new observation, which combines parameter and residual
    uncertainty). The test reports:

    - **Coverage:** the share of actual values inside their interval, with the shares below
    and above it, and the average interval width.
    - **Coverage by prediction bin:** coverage in `n_bins` quantile bins of the prediction.

    Two plots: coverage by bin against the nominal level, and actual values over the
    prediction band for a sample of `max_points` observations (seeded by `seed`), sorted by
    prediction.

    The test passes when overall coverage is at least `confidence_level -
    coverage_tolerance`.

    ### Signs of High Risk

    - Coverage well below the nominal level, especially out of time.
    - Misses concentrated on one side (all above, or all below): the model is biased, not
    just uncertain.
    - Bins with much lower coverage than others: the intervals are too narrow where the
    errors are largest (heteroscedasticity).

    ### Strengths

    - Tests the uncertainty statement directly, not the assumptions behind it.
    - Locates the failure by prediction range and by direction.

    ### Limitations

    - Requires a statsmodels OLS model.
    - On the data the model was fitted on, coverage is close to nominal by construction;
    use held-out or out-of-time data.
    - Coverage near nominal overall can hide over-coverage in some bins and under-coverage
    in others; read the bins.
    """
    results = ols_results(model, "PredictionIntervals")
    alpha = 1 - confidence_level

    # positions, not labels: a shuffled or duplicated index must not misalign rows
    df = dataset.df.reset_index(drop=True)
    if getattr(results.model, "formula", None) is not None:
        from patsy import dmatrix

        # drops rows with a missing regressor but, unlike get_prediction's own
        # transform, keeps the positions of the rows it keeps
        exog = dmatrix(results.model.data.design_info, df, return_type="dataframe")
        prediction = results.get_prediction(exog, transform=False)
    else:
        exog = design_data(results, df, dataset.feature_columns)
        prediction = results.get_prediction(exog)
    frame = prediction.summary_frame(alpha=alpha).set_axis(exog.index).reindex(df.index)

    columns = ["mean", "obs_ci_lower", "obs_ci_upper"]
    actual = np.asarray(dataset.y, dtype=float)
    # a NaN bound would count the row as inside the interval
    valid = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1) & np.isfinite(
        actual
    )
    if not valid.any():
        raise SkipTestError("PredictionIntervals found no rows without missing values")
    excluded = int((~valid).sum())
    predicted, lower, upper = (
        frame[column].to_numpy(dtype=float)[valid] for column in columns
    )
    actual = actual[valid]

    below, above = actual < lower, actual > upper
    inside = ~(below | above)
    coverage = float(inside.mean())

    coverage_df = pd.DataFrame(
        [
            {
                "Nominal Level": confidence_level,
                "Observations": len(actual),
                "Excluded (Missing)": excluded,
                "Coverage": coverage,
                "Below Interval": float(below.mean()),
                "Above Interval": float(above.mean()),
                "Mean Interval Width": float((upper - lower).mean()),
                "Pass/Fail": (
                    "Pass"
                    if coverage >= confidence_level - coverage_tolerance
                    else "Fail"
                ),
            }
        ]
    )

    bins = pd.qcut(predicted, q=n_bins, duplicates="drop")
    by_bin = (
        pd.DataFrame(
            {
                "bin": bins,
                "predicted": predicted,
                "inside": inside,
                "below": below,
                "above": above,
            }
        )
        .groupby("bin", observed=True)
        .agg(
            Count=("inside", "size"),
            Mean_Prediction=("predicted", "mean"),
            Coverage=("inside", "mean"),
            Below=("below", "mean"),
            Above=("above", "mean"),
        )
        .reset_index(drop=True)
        .rename(
            columns={
                "Mean_Prediction": "Mean Prediction",
                "Below": "Below Interval",
                "Above": "Above Interval",
            }
        )
    )

    coverage_fig = go.Figure(
        go.Bar(
            x=[f"{p:,.0f}" for p in by_bin["Mean Prediction"]],
            y=by_bin["Coverage"],
            marker_color=np.where(
                by_bin["Coverage"] >= confidence_level - coverage_tolerance,
                "#2e7d32",
                "#c62828",
            ),
        )
    )
    coverage_fig.add_hline(
        y=confidence_level,
        line_dash="dash",
        annotation_text=f"Nominal {confidence_level:.0%}",
    )
    coverage_fig.update_layout(
        title=f"Prediction Interval Coverage by Prediction Bin — {model.input_id} on {dataset.input_id}",
        xaxis_title="Mean prediction in bin",
        yaxis_title="Coverage",
        yaxis_range=[0, 1.05],
    )

    rng = np.random.default_rng(seed)
    shown = (
        np.sort(rng.choice(len(actual), max_points, replace=False))
        if len(actual) > max_points
        else np.arange(len(actual))
    )
    order = shown[np.argsort(predicted[shown])]
    rank = np.arange(len(order))
    band_fig = go.Figure()
    band_fig.add_trace(
        go.Scatter(
            x=np.concatenate([rank, rank[::-1]]),
            y=np.concatenate([upper[order], lower[order][::-1]]),
            fill="toself",
            fillcolor="rgba(92, 107, 192, 0.2)",
            line=dict(width=0),
            name=f"{confidence_level:.0%} prediction interval",
        )
    )
    band_fig.add_trace(
        go.Scatter(
            x=rank,
            y=predicted[order],
            mode="lines",
            line=dict(color="#3949ab"),
            name="Prediction",
        )
    )
    band_fig.add_trace(
        go.Scattergl(
            x=rank,
            y=actual[order],
            mode="markers",
            marker=dict(size=4, color=np.where(inside[order], "#757575", "#c62828")),
            name="Actual (red = outside)",
        )
    )
    band_fig.update_layout(
        title=f"Actual Values and Prediction Intervals — {model.input_id} on {dataset.input_id}",
        xaxis_title="Observations sorted by prediction",
        yaxis_title=dataset.target_column,
    )

    return (
        {
            "Prediction Interval Coverage": coverage_df.round(4),
            "Coverage by Prediction Bin": by_bin.round(4),
        },
        coverage_fig,
        band_fig,
        bool(coverage >= confidence_level - coverage_tolerance),
        RawData(
            lower=lower,
            upper=upper,
            predicted=predicted,
            excluded_missing=excluded,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
