# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("model_diagnosis", "conceptual_soundness", "visualization")
@tasks("regression")
def PredictionRange(
    dataset: VMDataset,
    model: VMModel,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    max_share_outside: float = 0.0,
) -> Tuple[pd.DataFrame, go.Figure, bool, RawData]:
    """
    Checks that a regression model's predictions stay within the range the target can
    take, and measures how many fall outside it.

    ### Purpose

    Many targets are bounded — a balance or a loss cannot be negative, a rate cannot exceed
    100%. A linear model has no such limits and will predict impossible values for some
    observations. Those predictions are conceptually unsound and, if used directly
    (summed into a portfolio forecast, compared with a limit), produce errors that fit
    metrics do not reveal.

    ### Test Mechanism

    The validator supplies `lower_bound`, `upper_bound` or both (at least one is required).
    For each bound the test counts the predictions beyond it, their share of the dataset,
    and the most extreme prediction. A histogram of predictions marks the bounds.

    The test passes when the share of predictions outside the bounds is at most
    `max_share_outside` (default 0: no impossible predictions).

    ### Signs of High Risk

    - A material share of predictions outside the feasible range.
    - Predictions far beyond a bound, not just marginally past it.

    ### Strengths

    - Works for any regression model with predictions.
    - Turns a domain constraint into an explicit, auditable check.

    ### Limitations

    - Only as good as the bounds supplied; it checks feasibility, not accuracy.
    - Flooring or capping predictions fixes the symptom but not the misspecification that
    caused it.
    """
    if lower_bound is None and upper_bound is None:
        raise ValueError("PredictionRange requires lower_bound, upper_bound or both")

    predicted = np.asarray(dataset.y_pred(model), dtype=float)
    n = len(predicted)

    rows = []
    outside = np.zeros(n, dtype=bool)
    for label, bound, beyond_fn, extreme in [
        ("Lower", lower_bound, np.less, predicted.min()),
        ("Upper", upper_bound, np.greater, predicted.max()),
    ]:
        if bound is None:
            continue
        beyond = beyond_fn(predicted, bound)
        outside |= beyond
        rows.append(
            {
                "Bound": label,
                "Value": bound,
                "Predictions Beyond": int(beyond.sum()),
                "Share Beyond": float(beyond.mean()),
                "Most Extreme Prediction": float(extreme),
            }
        )
    rows.append(
        {
            "Bound": "Total outside range",
            "Value": np.nan,
            "Predictions Beyond": int(outside.sum()),
            "Share Beyond": float(outside.mean()),
            "Most Extreme Prediction": np.nan,
        }
    )
    table = pd.DataFrame(rows)

    fig = go.Figure(
        go.Histogram(x=predicted, nbinsx=60, marker_color="#5c6bc0", name="Predictions")
    )
    for bound in (lower_bound, upper_bound):
        if bound is not None:
            fig.add_vline(x=bound, line_dash="dash", line_color="#c62828")
    fig.update_layout(
        title=f"Prediction Range — {model.input_id} on {dataset.input_id}",
        xaxis_title="Prediction",
        yaxis_title="Count",
    )

    passed = bool(outside.mean() <= max_share_outside)

    return (
        table.round(6),
        fig,
        passed,
        RawData(predicted=predicted, model=model.input_id, dataset=dataset.input_id),
    )
