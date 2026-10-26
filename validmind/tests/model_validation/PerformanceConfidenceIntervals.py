# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


def _metrics(actual, predicted):
    error = actual - predicted
    total = ((actual - actual.mean()) ** 2).sum()
    return {
        "R-Squared": 1 - (error**2).sum() / total,
        "RMSE": np.sqrt((error**2).mean()),
        "MAE": np.abs(error).mean(),
        "Mean Bias": error.mean(),
    }


_METRICS = ["R-Squared", "RMSE", "MAE", "Mean Bias"]

# For each compared metric: does a higher value mean a better model?
_HIGHER_IS_BETTER = {"R-Squared": True, "RMSE": False, "MAE": False}


def _difference_table(actual, predictions, draws, alpha, pct):
    """Each model minus the first, per metric, with paired-bootstrap intervals."""
    names = list(predictions)
    reference = names[0]
    reference_point = _metrics(actual, predictions[reference])
    rows = []
    for name in names[1:]:
        point = _metrics(actual, predictions[name])
        for metric, higher_is_better in _HIGHER_IS_BETTER.items():
            diff = draws[name][metric] - draws[reference][metric]
            difference = point[metric] - reference_point[metric]
            low, high = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
            significant = bool(low > 0 or high < 0)
            if not significant:
                better = "No significant difference"
            elif (difference > 0) == higher_is_better:
                better = name
            else:
                better = reference
            rows.append(
                {
                    "Comparison": f"{name} - {reference}",
                    "Metric": metric,
                    "Difference": difference,
                    f"{pct} CI Lower": low,
                    f"{pct} CI Upper": high,
                    "CI Excludes Zero": significant,
                    "Better Model": better,
                }
            )
    return pd.DataFrame(rows).round(4)


def _forest_plot(performance_df, pct, n_models):
    fig = make_subplots(rows=1, cols=len(_METRICS), subplot_titles=_METRICS)
    palette = ["#3949ab", "#d81b60", "#00897b", "#f9a825", "#6d4c41"]
    for k, metric in enumerate(_METRICS):
        rows = performance_df[performance_df["Metric"] == metric]
        fig.add_trace(
            go.Scatter(
                x=rows["Estimate"],
                y=rows["Model"],
                mode="markers",
                marker=dict(
                    size=10, color=[palette[i % len(palette)] for i in range(len(rows))]
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=rows[f"{pct} CI Upper"] - rows["Estimate"],
                    arrayminus=rows["Estimate"] - rows[f"{pct} CI Lower"],
                ),
                showlegend=False,
            ),
            row=1,
            col=k + 1,
        )
    fig.update_layout(height=200 + 60 * n_models)
    return fig


@tags("model_performance", "uncertainty", "model_comparison", "visualization")
@tasks("regression")
def PerformanceConfidenceIntervals(
    dataset: VMDataset,
    models: List[VMModel],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, RawData]:
    """
    Puts confidence intervals on regression performance metrics and, when several models
    are given, on the difference between each model and the first — so a comparison says
    whether one model is really better or just luckier on this sample.

    ### Purpose

    A test set is one sample. An R² of 0.80 against 0.78, or an MAE of 95 against 110, may
    or may not survive another sample; a champion/challenger decision taken on point
    estimates alone can replace a model for noise. Bootstrap intervals quantify that
    uncertainty, and the interval of the *difference* is what answers the comparison.

    ### Test Mechanism

    The rows of the dataset are resampled with replacement `n_bootstrap` times (seeded by
    `seed`). On every resample each model's R², RMSE, MAE and mean bias (`actual -
    predicted`) are computed from the predictions assigned to the dataset. Because all
    models are scored on the same resamples (a paired bootstrap), the differences account
    for the correlation between their errors.

    - **Performance:** each model's point estimate and percentile interval at
    `confidence_level`.
    - **Difference vs the first model:** for every other model, `model - first` for each
    metric, its interval, whether the interval excludes zero, and which model is better.
    Pass the reference model (for example the champion) first.

    A forest plot shows each metric's interval per model.

    ### Signs of High Risk

    - A challenger chosen on point estimates whose difference interval includes zero.
    - Wide intervals, meaning the dataset is too small to rank the models.
    - A mean bias interval that excludes zero: systematic over- or under-prediction.

    ### Strengths

    - Works for any regression models with predictions assigned to the dataset.
    - The paired design gives much tighter, and correct, intervals for differences than
    comparing two separate intervals.

    ### Limitations

    - Resampling rows assumes observations are independent.
    - Percentile intervals can be inaccurate for small datasets.
    - Quantifies sampling uncertainty only — not whether the dataset represents future
    data; use out-of-time data for that.
    """
    actual = np.asarray(dataset.y, dtype=float)
    predictions = {
        m.input_id: np.asarray(dataset.y_pred(m), dtype=float) for m in models
    }
    names = list(predictions)
    alpha = 1 - confidence_level

    rng = np.random.default_rng(seed)
    n = len(actual)
    draws = {name: [] for name in names}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        for name in names:
            draws[name].append(_metrics(actual[idx], predictions[name][idx]))
    draws = {name: pd.DataFrame(rows) for name, rows in draws.items()}

    pct = f"{confidence_level:.0%}"
    performance_rows = []
    for name in names:
        point = _metrics(actual, predictions[name])
        for metric, value in point.items():
            low, high = np.quantile(draws[name][metric], [alpha / 2, 1 - alpha / 2])
            performance_rows.append(
                {
                    "Model": name,
                    "Metric": metric,
                    "Estimate": value,
                    f"{pct} CI Lower": low,
                    f"{pct} CI Upper": high,
                }
            )
    performance_df = pd.DataFrame(performance_rows)

    tables = {"Performance with Confidence Intervals": performance_df.round(4)}

    if len(names) > 1:
        tables[f"Difference vs {names[0]}"] = _difference_table(
            actual, predictions, draws, alpha, pct
        )

    fig = _forest_plot(performance_df, pct, len(names))
    fig.update_layout(
        title=f"Performance with {pct} Bootstrap Confidence Intervals — {dataset.input_id}"
    )

    return (
        tables,
        fig,
        RawData(
            bootstrap_metrics=pd.concat(draws, names=["model", "draw"]),
            dataset=dataset.input_id,
        ),
    )
