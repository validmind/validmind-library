# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

from ._segments import mean_and_half_width, z_value


@tags("model_diagnosis", "calibration", "visualization")
@tasks("regression")
def CalibrationByDecile(
    dataset: VMDataset,
    model: VMModel,
    n_bins: int = 10,
    confidence_level: float = 0.95,
    adjust_for_multiple_bins: bool = True,
) -> Tuple[pd.DataFrame, go.Figure, bool, RawData]:
    """
    Checks that a regression model is right on average across the whole range of its
    predictions, by comparing mean actual and mean predicted values within prediction
    deciles.

    ### Purpose

    A well-calibrated model's predictions mean what they say: customers predicted at 500
    average about 500. Miscalibration at the low or high end — over-predicting small
    values, under-predicting large ones — is common for linear models of bounded or skewed
    targets, and matters because decisions are often taken at the extremes (the
    highest-exposure customers, the cases near a threshold).

    ### Test Mechanism

    Observations are sorted by prediction and split into `n_bins` equal-sized bins
    (deciles by default). For each bin the test reports the mean prediction, the mean
    actual value, and their difference with a confidence interval. A bin is **flagged**
    when its interval excludes zero; with `adjust_for_multiple_bins` (default) the
    confidence level is Bonferroni-adjusted for the number of bins. The plot shows mean
    actual against mean predicted per bin, with the diagonal of perfect calibration.

    The test passes when no bin is flagged.

    ### Signs of High Risk

    - Flagged bins at either end of the range.
    - A systematic S-shape or tilt in the plot: the model is too flat or too steep.
    - Mean predictions outside the feasible range in the lowest or highest bin.

    ### Strengths

    - Works for any regression model with predictions assigned to the dataset.
    - Shows where in the range the model is wrong, in the target's own units.

    ### Limitations

    - Calibration on average does not mean individual predictions are accurate.
    - On training data, OLS is calibrated overall by construction; use held-out data.
    """
    actual = np.asarray(dataset.y, dtype=float)
    predicted = np.asarray(dataset.y_pred(model), dtype=float)
    bins = pd.qcut(pd.Series(predicted).rank(method="first"), q=n_bins, labels=False)

    n_found = int(bins.nunique())
    z = z_value(confidence_level, n_found if adjust_for_multiple_bins else 1)

    rows = []
    for b in range(n_found):
        mask = (bins == b).to_numpy()
        gap, half_width = mean_and_half_width(actual[mask] - predicted[mask], z)
        rows.append(
            {
                "Bin": b + 1,
                "Count": int(mask.sum()),
                "Prediction Range": f"{predicted[mask].min():,.4g} – {predicted[mask].max():,.4g}",
                "Mean Predicted": float(predicted[mask].mean()),
                "Mean Actual": float(actual[mask].mean()),
                "Actual − Predicted": gap,
                "CI Lower": gap - half_width,
                "CI Upper": gap + half_width,
                "Result": "Miscalibrated" if abs(gap) > half_width else "OK",
            }
        )
    table = pd.DataFrame(rows)

    low = float(min(table["Mean Predicted"].min(), table["Mean Actual"].min()))
    high = float(max(table["Mean Predicted"].max(), table["Mean Actual"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line=dict(color="#757575", dash="dash"),
            name="Perfect calibration",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["Mean Predicted"],
            y=table["Mean Actual"],
            mode="lines+markers",
            marker=dict(
                size=9,
                color=np.where(table["Result"] == "OK", "#3949ab", "#c62828"),
            ),
            line=dict(color="#3949ab"),
            error_y=dict(
                type="data",
                array=table["CI Upper"] - table["Actual − Predicted"],
                arrayminus=table["Actual − Predicted"] - table["CI Lower"],
            ),
            name="Mean actual by prediction bin",
        )
    )
    fig.update_layout(
        title=f"Calibration by Prediction Bin — {model.input_id} on {dataset.input_id}",
        xaxis_title="Mean predicted",
        yaxis_title="Mean actual",
    )

    passed = bool((table["Result"] == "OK").all())

    return (
        table.round(4),
        fig,
        passed,
        RawData(
            bins=bins.to_numpy(),
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
