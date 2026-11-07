# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cecl_model import FEATURES
from validmind import tags, tasks
from validmind.vm_models import VMDataset

IV_STRENGTH = [
    (0.02, "not predictive"),
    (0.1, "weak"),
    (0.3, "medium"),
    (0.5, "strong"),
    (np.inf, "suspiciously strong"),
]


def _bins(values, n_bins):
    # Few distinct values (term, categoricals) are their own bins; otherwise quantiles.
    if values.dtype == object or values.nunique() <= n_bins:
        return values.astype(str)
    return pd.qcut(values, n_bins, duplicates="drop").astype(str)


@tags("cecl", "credit_risk", "feature_importance", "data_quality")
@tasks("classification")
def WOEInformationValue(
    dataset: VMDataset,
    target_column: str = None,
    features: list = None,
    n_bins: int = 10,
):
    """
    Measures each driver's Weight of Evidence (WoE) by bin and its Information Value (IV)
    against a binary event.

    ### Purpose

    WoE and IV are the standard credit-risk view of how strongly, and in what shape, each
    driver separates events from non-events on its own. They show whether a driver carries
    signal at all, whether the event rate moves across its range in the direction expected,
    and whether one driver is so strong that it suggests leakage. The model here is not a
    WoE scorecard, so the test documents the drivers rather than the model's inputs.

    ### Test Mechanism

    Each driver in `features` (default: the model's nine drivers) is binned: numeric drivers
    into `n_bins` quantile bins, drivers with `n_bins` or fewer distinct values into one bin
    per value. For each bin:

        WoE = ln(share of non-events in the bin / share of events in the bin)
        IV contribution = (share of non-events - share of events) * WoE

    Half an event and half a non-event are added to every bin so that a bin with no events
    has a finite WoE. The driver's IV is the sum over its bins, labelled from not predictive
    (< 0.02) to suspiciously strong (> 0.5). `target_column` selects the event and defaults
    to the dataset's target. The test is informational and has no pass/fail threshold.

    ### Signs of High Risk

    - A driver with an IV above 0.5, which can mean the driver leaks the outcome. The bands
      come from application scorecards; on a loan-quarter panel that spans a crisis, drivers
      that move with the cycle routinely exceed 0.5, so check the WoE pattern before
      concluding leakage
    - A driver the model relies on with an IV below 0.02
    - WoE that does not move steadily across a driver's range when the economics say it
      should, such as default risk that does not rise with LTV

    ### Strengths

    - Familiar to credit risk reviewers and comparable across drivers and samples
    - Implemented directly, with no binning library, so every number can be reproduced by hand

    ### Limitations

    - Univariate: a driver weak on its own can matter in combination, and correlated drivers
      each look strong
    - Pools loan-quarters across the whole sample, so drivers that act through the economic
      cycle (unemployment) mix level and timing effects
    - Quantile bins on a rare event can hold very few events; read WoE in sparse bins with care
    """
    target = target_column or dataset.target_column
    features = features or FEATURES
    df = dataset.df
    y = df[target].astype(int)

    rows = []
    for feature in features:
        g = y.groupby(_bins(df[feature], n_bins), sort=False)
        t = pd.DataFrame({"count": g.size(), "events": g.sum()})
        t["non_events"] = t["count"] - t["events"]
        t["event_rate"] = t["events"] / t["count"]
        share_events = (t["events"] + 0.5) / (t["events"].sum() + 0.5 * len(t))
        share_non_events = (t["non_events"] + 0.5) / (
            t["non_events"].sum() + 0.5 * len(t)
        )
        t["woe"] = np.log(share_non_events / share_events)
        t["iv"] = (share_non_events - share_events) * t["woe"]
        if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > n_bins:
            # Order quantile bins by their lower edge, not as text.
            t = t.iloc[np.argsort([float(b.split(",")[0][1:]) for b in t.index])]
        t.insert(0, "feature", feature)
        rows.append(t.reset_index(names="bin"))
    bins = pd.concat(rows, ignore_index=True)

    iv = bins.groupby("feature", sort=False)["iv"].sum().sort_values(ascending=False)
    summary = pd.DataFrame(
        {
            "feature": iv.index,
            "information_value": iv.to_numpy(),
            "strength": [
                next(label for edge, label in IV_STRENGTH if v < edge) for v in iv
            ],
        }
    )

    cols = 3
    rows_n = -(-len(features) // cols)
    fig = make_subplots(rows=rows_n, cols=cols, subplot_titles=list(iv.index))
    for k, feature in enumerate(iv.index):
        t = bins[bins["feature"] == feature]
        fig.add_trace(
            go.Bar(x=t["bin"], y=t["woe"], marker_color="#083E44"),
            row=k // cols + 1,
            col=k % cols + 1,
        )
    fig.update_layout(
        title=f"Weight of Evidence by bin ({target}, {dataset.input_id}) — "
        "negative = riskier than average",
        height=300 * rows_n,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)

    return (
        {
            "Information Value by driver": summary.round(4),
            "Weight of Evidence by bin": bins.round(4),
        },
        fig,
    )
