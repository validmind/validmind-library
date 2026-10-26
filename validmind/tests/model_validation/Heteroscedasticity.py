# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError, SkipTestError
from validmind.vm_models import VMDataset, VMModel

try:
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.tools.tools import add_constant
except ImportError as e:
    if "statsmodels" in str(e):
        raise MissingDependencyError(
            "Missing required package `statsmodels` for Heteroscedasticity. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["statsmodels"],
            extra="stats",
        ) from e

    raise e


@tags("model_diagnosis", "statistical_test", "visualization")
@tasks("regression")
def Heteroscedasticity(
    dataset: VMDataset,
    model: VMModel,
    significance_level: float = 0.05,
    n_bins: int = 10,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
    """
    Tests whether a regression model's error variance is constant, using the Breusch-Pagan
    and White tests and a residuals-versus-fitted plot.

    ### Purpose

    Ordinary least squares assumes every observation has the same error variance. When the
    variance changes with the regressors or the prediction (heteroscedasticity), the
    coefficients remain unbiased but their standard errors, p-values and prediction
    intervals are wrong — usually too optimistic where the errors are largest. It also tells
    a business reader the model is much less reliable for some customers than for others.

    ### Test Mechanism

    Residuals are `actual - predicted` on the dataset, using the predictions assigned to it
    for the model. The squared residuals are regressed on the dataset's numeric features:

    - **Breusch-Pagan:** tests for variance that changes linearly with the features.
    - **White:** adds squares and cross-products of the features, so it also detects
    non-linear patterns.

    A p-value below `significance_level` rejects constant variance. The plot shows residuals
    against predictions, with the residual standard deviation in `n_bins` bins of the
    prediction overlaid; a constant-variance model gives a flat band.

    The test passes when neither test rejects.

    ### Signs of High Risk

    - Rejection by both tests.
    - A funnel or bow shape in the residuals-versus-fitted plot, or a residual standard
    deviation that changes several-fold across prediction bins.

    ### Strengths

    - Works for any regression model with predictions.
    - Pairs formal tests with a plot that shows where the variance changes.

    ### Limitations

    - Uses the dataset's numeric feature columns as the explanatory variables; categorical
    features are not included, and variance driven only by them is not tested.
    - The White test uses many terms when there are many features, which lowers its power.
    - In large samples both tests reject small, immaterial variance changes; use the binned
    standard deviations to judge materiality.
    """
    actual = np.asarray(dataset.y, dtype=float)
    predicted = np.asarray(dataset.y_pred(model), dtype=float)
    residuals = actual - predicted

    numeric = list(dataset.feature_columns_numeric)
    if not numeric:
        raise SkipTestError("Heteroscedasticity requires numeric feature columns")

    regressors = add_constant(
        dataset.df[numeric].to_numpy(dtype=float), has_constant="add"
    )

    rows = []
    for name, test in [("Breusch-Pagan", het_breuschpagan), ("White", het_white)]:
        lm, lm_p, _, _ = test(residuals, regressors)
        rows.append(
            {
                "Test": name,
                "LM Statistic": float(lm),
                "p-value": float(lm_p),
                "Significance Level": significance_level,
                "Pass/Fail": "Pass" if lm_p >= significance_level else "Fail",
            }
        )
    tests_df = pd.DataFrame(rows)

    bins = pd.qcut(predicted, q=n_bins, duplicates="drop")
    binned = (
        pd.DataFrame({"bin": bins, "predicted": predicted, "residual": residuals})
        .groupby("bin", observed=True)
        .agg(
            Count=("residual", "size"),
            Mean_Prediction=("predicted", "mean"),
            Residual_Std=("residual", "std"),
        )
        .reset_index(drop=True)
        .rename(
            columns={
                "Mean_Prediction": "Mean Prediction",
                "Residual_Std": "Residual Std. Deviation",
            }
        )
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=predicted,
            y=residuals,
            mode="markers",
            marker=dict(size=3, opacity=0.35, color="#5c6bc0"),
            name="Residuals",
        )
    )
    for sign in (1, -1):
        fig.add_trace(
            go.Scatter(
                x=binned["Mean Prediction"],
                y=sign * binned["Residual Std. Deviation"],
                mode="lines+markers",
                line=dict(color="#d81b60"),
                name="± residual std. deviation by bin",
                showlegend=sign == 1,
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#757575")
    fig.update_layout(
        title=f"Residuals vs Predicted — {model.input_id} on {dataset.input_id}",
        xaxis_title="Predicted",
        yaxis_title="Residual",
    )

    passed = bool((tests_df["Pass/Fail"] == "Pass").all())

    return (
        {
            "Heteroscedasticity Tests": tests_df.round(6),
            "Residual Spread by Prediction Bin": binned.round(4),
        },
        fig,
        passed,
        RawData(
            residuals=residuals,
            predicted=predicted,
            model=model.input_id,
            dataset=dataset.input_id,
        ),
    )
