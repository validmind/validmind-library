# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.vm_models import VMDataset, VMModel

from .statsmodels.statsutils import predict_frame

_OPERATIONS = {
    "multiply": lambda values, x: values * x,
    "add": lambda values, x: values + x,
    "set": lambda values, x: np.full(len(values), x),
}


def _apply(df, scenario_name, changes):
    frame = df.copy()
    for feature, change in changes.items():
        if feature not in frame.columns:
            raise ValueError(f"Scenario {scenario_name!r}: unknown feature {feature!r}")
        if not isinstance(change, dict) or len(change) != 1:
            raise ValueError(
                f"Scenario {scenario_name!r}, feature {feature!r}: give exactly one of "
                f"{sorted(_OPERATIONS)}, for example {{'multiply': 0.8}}"
            )
        ((operation, amount),) = change.items()
        if operation not in _OPERATIONS:
            raise ValueError(
                f"Scenario {scenario_name!r}, feature {feature!r}: unknown operation "
                f"{operation!r}; use one of {sorted(_OPERATIONS)}"
            )
        frame[feature] = _OPERATIONS[operation](frame[feature].to_numpy(), amount)
    return frame


@tags("sensitivity_analysis", "stress_testing", "visualization")
@tasks("regression")
def ScenarioAnalysis(
    dataset: VMDataset,
    model: VMModel,
    scenarios: Dict[str, Dict[str, Dict[str, Any]]],
    lower_bound: Optional[float] = None,
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Projects a regression model's portfolio-level prediction under named stress scenarios
    in which several features move together.

    ### Purpose

    Stress testing asks what the model forecasts if conditions change — a recession that
    cuts incomes and raises debt burdens, a policy that raises credit limits. It checks that
    the model's response is plausible in direction and size, and gives the business the
    numbers it needs for planning. Unlike one-at-a-time sensitivity, scenarios move features
    jointly, as they move in reality.

    ### Test Mechanism

    The validator defines `scenarios`, each a name mapped to feature changes. Each change
    is one operation applied to every row:

    - `{"multiply": 0.8}` — scale the feature (income down 20%),
    - `{"add": 0.05}` — shift it (debt-to-income up 5 points),
    - `{"set": 1}` — set it to a value.

    For example `{"Recession": {"Income": {"multiply": 0.8}, "DebtToIncome": {"add":
    0.05}}}`. The model predicts on the baseline dataset and on each scenario's copy; the
    test reports the mean and total prediction, and the change in the total from the
    baseline in units and percent. With `lower_bound`, it also reports the share of
    predictions below it, since a stress can push a linear model into impossible values.

    ### Signs of High Risk

    - A scenario whose impact has an implausible direction or size.
    - A stress that pushes a material share of predictions outside the feasible range.

    ### Strengths

    - Works for any regression model that can predict on the dataset's features.
    - Scenarios are explicit, named and reproducible, so results can be reviewed and
    compared across model versions.

    ### Limitations

    - Only as good as the scenarios supplied; it applies them mechanically and does not
    model second-round effects (for example customers leaving the portfolio).
    - Applies the same change to every row.
    """
    feature_columns = dataset.feature_columns
    frames = {"Baseline": dataset.df}
    for name, changes in scenarios.items():
        frames[name] = _apply(dataset.df, name, changes)

    predictions = {
        name: predict_frame(model, frame, feature_columns)
        for name, frame in frames.items()
    }
    baseline_total = predictions["Baseline"].sum()

    rows = []
    for name, predicted in predictions.items():
        row = {
            "Scenario": name,
            "Changes": (
                "; ".join(
                    f"{feature} {op} {amount}"
                    for feature, change in scenarios[name].items()
                    for op, amount in change.items()
                )
                if name in scenarios
                else "—"
            ),
            "Mean Prediction": predicted.mean(),
            "Total Prediction": predicted.sum(),
            "Change in Total": predicted.sum() - baseline_total,
            "% Change in Total": (predicted.sum() - baseline_total)
            / abs(baseline_total),
        }
        if lower_bound is not None:
            row[f"Share Below {lower_bound}"] = float((predicted < lower_bound).mean())
        rows.append(row)
    table = pd.DataFrame(rows)

    stressed = table[table["Scenario"] != "Baseline"]
    fig = go.Figure(
        go.Bar(
            x=stressed["Scenario"],
            y=stressed["% Change in Total"],
            marker_color=np.where(
                stressed["% Change in Total"] >= 0, "#3949ab", "#d81b60"
            ),
            text=[f"{v:+.1%}" for v in stressed["% Change in Total"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Scenario Impact on Total Prediction — {model.input_id} on {dataset.input_id}",
        yaxis_title="% change in total prediction vs baseline",
        yaxis_tickformat=".0%",
    )

    return (
        table.round(4),
        fig,
        RawData(
            predictions=predictions, model=model.input_id, dataset=dataset.input_id
        ),
    )
