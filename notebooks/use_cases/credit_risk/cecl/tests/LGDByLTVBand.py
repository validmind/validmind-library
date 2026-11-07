# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd
import plotly.graph_objects as go

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("cecl", "credit_risk", "lgd", "backtesting")
@tasks("regression")
def LGDByLTVBand(
    datasets: list[VMDataset],
    model: VMModel,
    bands: list = None,
    max_abs_error: float = 0.05,
    min_count: int = 20,
):
    """
    Compares realized and predicted loss given default within mark-to-market LTV bands.

    ### Purpose

    For mortgages, severity is driven by the collateral: the deeper a borrower's negative equity
    at default, the larger the shortfall after the house is sold. A CECL allowance applies the LGD
    model at **projected** loan-to-values, which under a house-price shock move into the high
    bands where defaults are scarce. The model therefore has to be right band by band, not just on
    the average severity, which a low-LTV-heavy sample dominates.

    ### Test Mechanism

    Each dataset in `datasets` is a set of defaults with the LGD model's predictions assigned,
    typically the development and out-of-time defaults. The test bins defaults by current LTV
    (`bands`) and reports the count, mean realized LGD, mean predicted LGD and their difference
    per band and dataset. The test passes when every band with at least `min_count` defaults has
    an absolute error within `max_abs_error`.

    ### Signs of High Risk

    - Under-prediction in the bands above 90% LTV, which understates stress losses
    - Errors that grow out-of-time, pointing to a change in recovery conditions (foreclosure
      timelines, sale discounts) the model does not capture
    - High-LTV bands with too few defaults to test at all, so the model's behaviour there rests
      on extrapolation

    ### Strengths

    - Tests the LGD model along the dimension the scenarios stress
    - Separates calibration in sparse, high-impact bands from the dense, low-impact bulk

    ### Limitations

    - Band means hide dispersion; a correct mean can coexist with poor loan-level accuracy
    - Thin high-LTV bands fall below `min_count` and are reported but not judged
    """
    bands = bands or [0, 0.6, 0.8, 0.9, 1.0, 1.2, 5]
    rows = []
    for ds in datasets:
        frame = pd.DataFrame(
            {
                "band": pd.cut(ds.df["current_ltv"], bands).astype(str),
                "actual": ds.df[ds.target_column].to_numpy(),
                "predicted": ds.y_pred(model),
            }
        )
        g = frame.groupby("band", sort=False)
        t = pd.DataFrame(
            {
                "defaults": g.size(),
                "actual_mean_lgd": g["actual"].mean(),
                "predicted_mean_lgd": g["predicted"].mean(),
            }
        )
        t["error"] = t["predicted_mean_lgd"] - t["actual_mean_lgd"]
        t["judged"] = t["defaults"] >= min_count
        t["within_tolerance"] = ~t["judged"] | (t["error"].abs() <= max_abs_error)
        t.insert(0, "dataset", ds.input_id)
        rows.append(t.reset_index())

    table = pd.concat(rows, ignore_index=True)
    order = [str(b) for b in pd.cut(pd.Series([]), bands).cat.categories]
    table["band"] = pd.Categorical(table["band"], categories=order)
    table = table.sort_values(["dataset", "band"])

    fig = go.Figure()
    for ds_id, colors in zip(
        table["dataset"].unique(), [("#083E44", "#6FA8B0"), ("#DE257E", "#F2A541")]
    ):
        t = table[table["dataset"] == ds_id]
        fig.add_trace(
            go.Bar(
                x=t["band"].astype(str),
                y=t["actual_mean_lgd"],
                name=f"{ds_id} actual",
                marker_color=colors[0],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t["band"].astype(str),
                y=t["predicted_mean_lgd"],
                name=f"{ds_id} predicted",
                mode="lines+markers",
                line_color=colors[1],
            )
        )
    fig.update_layout(
        title="Mean LGD by current LTV band",
        xaxis_title="current LTV",
        yaxis_title="LGD",
        barmode="group",
        height=400,
    )

    return (
        {"LGD by LTV band": table.round(4)},
        fig,
        bool(table["within_tolerance"].all()),
    )
