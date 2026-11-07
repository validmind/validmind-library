# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel


@tags("cecl", "credit_risk", "calibration", "time_series")
@tasks("classification")
def QuarterlyCalibration(
    dataset: VMDataset,
    model: VMModel,
    target_column: str = None,
    confidence: float = 0.95,
    max_breach_share: float = 0.10,
):
    """
    Compares predicted and observed event rates quarter by quarter, with a binomial confidence band.

    ### Purpose

    A CECL hazard model is used to forecast losses through the economic cycle, so it has to be
    calibrated in each quarter, not only on average. A model can match the pooled event rate while
    systematically under-predicting in downturns and over-predicting in benign periods, which
    understates the allowance exactly when it matters. This test checks the event rate the model
    predicts against the one observed in every calendar quarter.

    ### Test Mechanism

    For each quarter the test sums the predicted probabilities to get the expected number of events
    and compares it with the observed count. Under the model, the observed count has variance
    $\\sum p_i(1-p_i)$, giving a z-score per quarter. A quarter is a **breach** when its z-score is
    outside the two-sided band at the chosen `confidence`. The test passes when no more than
    `max_breach_share` of quarters breach.

    `target_column` selects the event: it defaults to the dataset's target, so the same test checks
    the default model on `defaulted` and the prepayment model on `prepaid`.

    ### Signs of High Risk

    - Breaches clustered in consecutive quarters, which point to a missing driver or a regime
      change rather than noise
    - Over-prediction in stress quarters the model has not seen, such as policy-driven periods
      (forbearance) where the usual macro relationship broke down
    - Persistent under-prediction in recent quarters, which understates the current allowance

    ### Strengths

    - Uses the binomial variance of the model's own predictions, so small and large quarters are
      judged on the same footing
    - Shows calibration through time, which pooled metrics such as AUC or a single A/E ratio hide

    ### Limitations

    - Treats loan-quarters as independent; correlated defaults within a quarter make the band too
      narrow, so breaches should be read alongside their size
    - Diagnoses where calibration fails, not why
    """
    df = dataset.df
    target = target_column or dataset.target_column
    frame = pd.DataFrame(
        {
            "quarter": df["quarter"].astype(str),
            "actual": df[target].to_numpy(),
            "p": dataset.y_prob(model),
        }
    )
    frame["var"] = frame["p"] * (1 - frame["p"])
    g = frame.groupby("quarter")
    table = pd.DataFrame(
        {
            "loan_quarters": g.size(),
            "actual_events": g["actual"].sum(),
            "expected_events": g["p"].sum(),
            "std": np.sqrt(g["var"].sum()),
        }
    )
    z_crit = norm.ppf(0.5 + confidence / 2)
    table["z_score"] = (table["actual_events"] - table["expected_events"]) / table[
        "std"
    ]
    table["breach"] = table["z_score"].abs() > z_crit
    table["actual_rate_annualized"] = (
        table["actual_events"] / table["loan_quarters"] * 4
    )
    table["predicted_rate_annualized"] = (
        table["expected_events"] / table["loan_quarters"] * 4
    )
    band = z_crit * table["std"] / table["loan_quarters"] * 4

    x = pd.PeriodIndex(table.index, freq="Q").to_timestamp()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.r_[x, x[::-1]],
            y=np.r_[
                (table["predicted_rate_annualized"] + band) * 100,
                ((table["predicted_rate_annualized"] - band).clip(lower=0) * 100)[::-1],
            ],
            fill="toself",
            fillcolor="rgba(222,37,126,0.12)",
            line_width=0,
            name=f"{confidence:.0%} band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=table["predicted_rate_annualized"] * 100,
            name="predicted",
            line_color="#DE257E",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=table["actual_rate_annualized"] * 100,
            name="actual",
            line_color="#083E44",
        )
    )
    breaches = table[table["breach"]]
    fig.add_trace(
        go.Scatter(
            x=pd.PeriodIndex(breaches.index, freq="Q").to_timestamp(),
            y=breaches["actual_rate_annualized"] * 100,
            mode="markers",
            marker=dict(color="#F2A541", size=9, symbol="x"),
            name="breach",
        )
    )
    fig.update_layout(
        title=f"{target}: actual vs predicted rate by quarter ({dataset.input_id})",
        yaxis_title="% annualized",
        height=420,
    )

    breach_share = table["breach"].mean()
    summary = pd.DataFrame(
        {
            "quarters": [len(table)],
            "breaches": [int(table["breach"].sum())],
            "breach_share": [breach_share],
            "max_breach_share": [max_breach_share],
            "overall_actual_to_expected": [
                table["actual_events"].sum() / table["expected_events"].sum()
            ],
        }
    )
    return (
        {
            "Summary": summary,
            "Calibration by quarter": table.reset_index().drop(columns="std").round(4),
        },
        fig,
        bool(breach_share <= max_breach_share),
    )
