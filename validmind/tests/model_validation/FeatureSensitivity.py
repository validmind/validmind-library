# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMDataset, VMModel

from .statsmodels.statsutils import predict_frame


def _is_binary(values):
    return set(np.unique(values)) <= {0, 1}


@tags("sensitivity_analysis", "visualization")
@tasks("regression")
def FeatureSensitivity(
    dataset: VMDataset,
    model: VMModel,
    features: Optional[List[str]] = None,
    shock_std: float = 1.0,
    shocks: Optional[Dict[str, float]] = None,
    clip_to_range: bool = True,
) -> Tuple[pd.DataFrame, go.Figure, RawData]:
    """
    Measures how a regression model's portfolio-level prediction responds when each feature
    is shocked up and down, one at a time.

    ### Purpose

    Sensitivity analysis shows which inputs drive the model's output and by how much, in
    business units: "a one-standard-deviation rise in income lowers the average predicted
    balance by $40". It reveals features whose influence is larger than their business
    importance would justify, effects that are asymmetric (a non-linear model reacting
    differently to rises and falls), and features that barely matter.

    ### Test Mechanism

    For each feature (by default every numeric feature of the dataset), two copies of the
    dataset are made with only that feature changed:

    - **Continuous features:** shifted up and down by `shock_std` standard deviations, or by
    the amount given for that feature in `shocks` (in the feature's own units).
    - **Binary (0/1) features:** set to 1 for everyone (up) and to 0 for everyone (down).

    With `clip_to_range`, shocked values are kept within the feature's observed range, so
    no shock creates an impossible value (for example a negative credit limit).

    The model predicts on each copy, and the test reports the mean prediction under each
    shock and its change from the baseline, in units and in percent. A tornado chart ranks
    the features by the size of their effect.

    ### Signs of High Risk

    - A feature with a much larger effect than business knowledge supports.
    - An effect with the wrong direction.
    - Large asymmetry between up and down shocks that the model's purpose does not explain.

    ### Strengths

    - Works for any regression model that can predict on the dataset's features.
    - Effects are expressed in the target's units at portfolio level, which is how the
    model's output is used.

    ### Limitations

    - One feature at a time: correlated features would move together in reality; use
    scenario analysis for joint movements.
    - Categorical (non-numeric) features are not shocked.
    - The baseline is the dataset's own predictions recomputed by the model; models that
    cannot predict on the dataset's features are not supported.
    """
    df = dataset.df
    feature_columns = dataset.feature_columns
    features = features or list(dataset.feature_columns_numeric)
    shocks = shocks or {}
    if not features:
        raise SkipTestError("FeatureSensitivity requires numeric features")

    baseline = predict_frame(model, df, feature_columns).mean()

    rows = []
    for feature in features:
        values = df[feature].to_numpy(dtype=float)
        if _is_binary(values):
            up, down, shock_label = np.ones_like(values), np.zeros_like(values), "0 → 1"
        else:
            amount = shocks.get(feature, shock_std * values.std(ddof=1))
            up, down = values + amount, values - amount
            if clip_to_range:
                up = np.clip(up, values.min(), values.max())
                down = np.clip(down, values.min(), values.max())
            shock_label = f"± {amount:,.4g}"

        means = {}
        for direction, shocked in (("Up", up), ("Down", down)):
            frame = df.copy()
            frame[feature] = shocked
            means[direction] = predict_frame(model, frame, feature_columns).mean()

        rows.append(
            {
                "Feature": feature,
                "Shock": shock_label,
                "Baseline Mean Prediction": baseline,
                "Mean Prediction (Up)": means["Up"],
                "Mean Prediction (Down)": means["Down"],
                "Change (Up)": means["Up"] - baseline,
                "Change (Down)": means["Down"] - baseline,
                "% Change (Up)": (means["Up"] - baseline) / abs(baseline),
                "% Change (Down)": (means["Down"] - baseline) / abs(baseline),
            }
        )

    table = pd.DataFrame(rows)
    table["Range"] = (table["Change (Up)"] - table["Change (Down)"]).abs()
    table = table.sort_values("Range", ascending=False).reset_index(drop=True)

    ordered = table.iloc[::-1]
    fig = go.Figure()
    for direction, color in (("Up", "#3949ab"), ("Down", "#d81b60")):
        fig.add_trace(
            go.Bar(
                y=ordered["Feature"],
                x=ordered[f"Change ({direction})"],
                orientation="h",
                name=f"Shock {direction.lower()}",
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=f"Feature Sensitivity — {model.input_id} on {dataset.input_id}",
        xaxis_title=f"Change in mean prediction (baseline {baseline:,.4g})",
        height=200 + 35 * len(table),
    )

    return (
        table.drop(columns="Range").round(4),
        fig,
        RawData(sensitivity=table, model=model.input_id, dataset=dataset.input_id),
    )
