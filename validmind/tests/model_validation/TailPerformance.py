# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("model_performance", "segment_analysis", "visualization")
@tasks("regression")
def TailPerformance(
    dataset: VMDataset,
    model: VMModel,
    tail_share: float = 0.1,
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Measures a regression model's accuracy and bias in the tails — the observations with
    the largest and smallest actual values, and those with the largest and smallest
    predictions — and compares them with the whole dataset.

    ### Purpose

    Exposure, loss and value are usually concentrated in the tails: the largest balances,
    the most severe losses. A model can be accurate for typical observations and badly
    wrong where the money is. Looking at the tails by actual value shows how the model
    treats extreme outcomes; looking at them by prediction shows how reliable its most
    extreme forecasts are — the ones most likely to trigger a decision.

    ### Test Mechanism

    Five groups are compared: all observations; the top and bottom `tail_share` (10% by
    default) by actual value; and the top and bottom `tail_share` by prediction, with ties
    broken by row order so each tail holds exactly that share. For each
    group the test reports the count, the mean actual and predicted values, the mean
    residual (`actual - predicted`), the MAE, and the MAE relative to the overall MAE. A
    bar chart shows the mean residual by group.

    ### Signs of High Risk

    - Large under-prediction in the top tail by actual value: the model misses the extreme
    outcomes that matter most.
    - Large errors in the top tail by prediction: the model's highest forecasts are not
    reliable.
    - Predictions in the bottom tail outside the feasible range.

    ### Strengths

    - Works for any regression model with predictions assigned to the dataset.
    - Separates "extreme outcomes" from "extreme forecasts", which fail differently.

    ### Limitations

    - Some regression to the mean is expected: any imperfect model under-predicts the top
    tail by actual value and over-predicts the bottom one. Judge the size, not the sign.
    - Tails defined by a share hold few observations in small datasets.
    """
    actual = np.asarray(dataset.y, dtype=float)
    predicted = np.asarray(dataset.y_pred(model), dtype=float)
    residuals = actual - predicted
    overall_mae = float(np.abs(residuals).mean())

    # rank rather than compare with a quantile, so ties (such as many zero balances)
    # cannot push a tail beyond its share
    def ranks(values):
        return pd.Series(values).rank(method="first").to_numpy()

    def top(values):
        return ranks(values) > len(values) * (1 - tail_share)

    def bottom(values):
        return ranks(values) <= len(values) * tail_share

    pct = f"{tail_share:.0%}"
    groups = {
        "All observations": np.ones(len(actual), dtype=bool),
        f"Top {pct} by actual": top(actual),
        f"Bottom {pct} by actual": bottom(actual),
        f"Top {pct} by prediction": top(predicted),
        f"Bottom {pct} by prediction": bottom(predicted),
    }

    rows = []
    for group, mask in groups.items():
        mae = float(np.abs(residuals[mask]).mean())
        rows.append(
            {
                "Group": group,
                "Count": int(mask.sum()),
                "Mean Actual": float(actual[mask].mean()),
                "Mean Predicted": float(predicted[mask].mean()),
                "Mean Residual": float(residuals[mask].mean()),
                "MAE": mae,
                "MAE Ratio to Overall": mae / overall_mae if overall_mae else np.nan,
            }
        )
    table = pd.DataFrame(rows)

    fig = go.Figure(
        go.Bar(
            x=table["Group"],
            y=table["Mean Residual"],
            marker_color=np.where(table["Mean Residual"] >= 0, "#3949ab", "#d81b60"),
            text=[f"{v:,.4g}" for v in table["Mean Residual"]],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="#757575")
    fig.update_layout(
        title=(
            f"Mean Residual in the Tails (positive = under-prediction) — "
            f"{model.input_id} on {dataset.input_id}"
        ),
        yaxis_title="Mean residual (actual − predicted)",
    )

    return (
        table.round(4),
        fig,
        RawData(residuals=residuals, model=model.input_id, dataset=dataset.input_id),
    )
