# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cecl_model import model_features
from validmind import tags, tasks
from validmind.vm_models import VMDataset, VMModel

NUMERIC_DRIVERS = [
    "fico",
    "current_ltv",
    "dti",
    "loan_age",
    "unemployment_rate",
    "refi_incentive",
]


@tags("cecl", "credit_risk", "sensitivity", "model_explainability")
@tasks("classification")
def DriverSensitivity(
    dataset: VMDataset,
    model: VMModel,
    expected_direction: dict = None,
    sample_size: int = 5000,
    grid_points: int = 30,
    extend: float = 0.5,
    seed: int = 42,
):
    """
    Traces the model's response to each risk driver across and beyond the range it was trained on.

    ### Purpose

    A CECL model is run under stress scenarios, so its response to each driver has to make
    economic sense: default risk must not fall as unemployment or loan-to-value rises. The
    response must also keep moving when a scenario pushes a driver past anything in the
    development data. Tree-based models are flat outside their training range, so a severe
    scenario can produce the same loss as a moderate one without any warning. This test shows both.

    ### Test Mechanism

    For each driver the test computes a partial-dependence curve. It sets the driver to each value
    on a grid for a random sample of `sample_size` loan-quarters from the dataset, leaves every
    other feature as observed, and averages the model's predicted probability. The grid spans the
    observed 1st–99th percentile range, widened by `extend` times that range on each side. The
    curve is then compared with `expected_direction` (`1` increasing, `-1` decreasing) for the
    drivers listed there. The test passes when every listed driver moves only in its expected
    direction.

    It also reports how much the prediction changes **beyond** the observed range compared with
    within it. A ratio near zero means the model will not respond to scenarios more extreme than
    its development data.

    ### Signs of High Risk

    - A driver moving against its expected sign anywhere in the grid
    - A response that stops changing just past the observed maximum of a macro driver, so stress
      scenarios are capped at the worst conditions in the development window
    - Steep, jagged curves inside the observed range, which point to overfitting

    ### Strengths

    - Model-agnostic: it reads only predictions, so it works the same for boosted trees, neural
      networks or regressions
    - Makes extrapolation behaviour visible, which accuracy metrics cannot show

    ### Limitations

    - Partial dependence averages over the sample, so it can hide interactions that reverse the
      sign for a sub-population
    - Setting a driver independently of the others can create combinations that never occur
      together, such as high unemployment with a low loan-to-value
    """
    rng = np.random.default_rng(seed)
    df = dataset.df
    sample = df.iloc[rng.choice(len(df), min(sample_size, len(df)), replace=False)]
    X = model_features(sample)
    expected_direction = expected_direction or {}
    estimator = model.model

    rows, curves = [], {}
    for driver in NUMERIC_DRIVERS:
        lo, hi = df[driver].quantile([0.01, 0.99])
        observed_min, observed_max = df[driver].min(), df[driver].max()
        span = hi - lo
        grid = np.linspace(lo - extend * span, hi + extend * span, grid_points)
        if driver in ("fico", "current_ltv", "dti", "loan_age"):
            grid = grid[grid >= 0]
        response = []
        for value in grid:
            Xv = X.copy()
            Xv[driver] = value
            response.append(estimator.predict_proba(Xv)[:, 1].mean())
        response = np.array(response)
        curves[driver] = (grid, response, observed_min, observed_max)

        steps = np.diff(response)
        tolerance = 1e-9 * max(response.max(), 1e-12)
        direction = expected_direction.get(driver)
        if direction:
            monotonic = bool((steps * direction >= -tolerance).all())
        else:
            monotonic = None
        inside = (grid >= observed_min) & (grid <= observed_max)
        within = response[inside].max() - response[inside].min() if inside.any() else 0
        above = grid > observed_max
        beyond = (
            abs(response[above][-1] - response[inside][-1])
            if above.any() and inside.any()
            else np.nan
        )
        rows.append(
            {
                "driver": driver,
                "expected_direction": direction or "not set",
                "observed_min": observed_min,
                "observed_max": observed_max,
                "monotonic_as_expected": monotonic,
                "change_within_range": within,
                "change_beyond_max": beyond,
                "beyond_to_within_ratio": beyond / within if within else np.nan,
            }
        )

    fig = make_subplots(rows=2, cols=3, subplot_titles=list(curves))
    for i, (driver, (grid, response, observed_min, observed_max)) in enumerate(
        curves.items()
    ):
        r, c = divmod(i, 3)
        fig.add_trace(
            go.Scatter(
                x=grid, y=response * 100, line_color="#083E44", showlegend=False
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.add_vrect(
            x0=observed_min,
            x1=observed_max,
            fillcolor="grey",
            opacity=0.08,
            line_width=0,
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        title=f"{model.input_id}: mean predicted probability (%) by driver — shaded = observed range",
        height=560,
    )

    table = pd.DataFrame(rows)
    checked = table[table["monotonic_as_expected"].notna()]
    return (
        {"Driver sensitivity": table.round(6)},
        fig,
        bool(checked["monotonic_as_expected"].all()),
    )
