# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import SkipTestError
from validmind.vm_models import VMModel


@tags("tabular_data", "visualization", "model_training")
@tasks("regression")
def RegressionCoeffs(
    model: VMModel, confidence_level: float = 0.95
) -> Tuple[go.Figure, RawData, pd.DataFrame]:
    """
    Assesses the significance and uncertainty of predictor variables in a regression model through visualization of
    coefficients and their confidence intervals.

    ### Purpose

    The `RegressionCoeffs` metric visualizes the estimated regression coefficients alongside their confidence intervals (95% by default),
    providing insights into the impact and significance of predictor variables on the response variable. This visualization
    helps to understand the variability and uncertainty in the model's estimates, aiding in the evaluation of the
    significance of each predictor.

    ### Test Mechanism

    The function operates by extracting the estimated coefficients and their confidence intervals at `confidence_level`
    (default 95%) from the fitted statsmodels model, which uses the model's own reference distribution (Student's t for
    OLS). The confidence interval indicates the range within which the true coefficient value is expected to fall with
    that confidence. The coefficients and their confidence intervals are then visualized
    in a bar plot. The x-axis represents the predictor variables, the y-axis represents the estimated coefficients, and
    the error bars depict the confidence intervals.

    ### Signs of High Risk

    - The confidence interval for a coefficient contains the zero value, suggesting that the predictor may not significantly
    contribute to the model.
    - Multiple coefficients with confidence intervals that include zero, potentially indicating issues with model reliability.
    - Very wide confidence intervals, which may suggest high uncertainty in the coefficient estimates and potential model
    instability.

    ### Strengths

    - Provides a clear visualization that allows for easy interpretation of the significance and impact of predictor
    variables.
    - Includes confidence intervals, which provide additional information about the uncertainty surrounding each coefficient
    estimate.

    ### Limitations

    - The method assumes normality of residuals and independence of observations, assumptions that may not always hold true
    in practice.
    - It does not address issues related to multi-collinearity among predictor variables, which can affect the interpretation
    of coefficients.
    - This metric is limited to regression tasks using tabular data and is not applicable to other types of machine learning
    tasks or data structures.
    """

    if model.library != "statsmodels":
        raise SkipTestError("Only statsmodels are supported for this metric")

    coefficients = model.regression_coefficients()
    coef = pd.Series(model.model.params).to_numpy(dtype=float)
    std_err = pd.Series(model.model.bse).to_numpy(dtype=float)
    intervals = pd.DataFrame(model.model.conf_int(alpha=1 - confidence_level))
    lower_ci = intervals.iloc[:, 0].to_numpy(dtype=float)
    upper_ci = intervals.iloc[:, 1].to_numpy(dtype=float)

    # the summary table's interval columns are always 95%; show the chart's level instead
    alpha = 1 - confidence_level
    coefficients = coefficients.drop(columns=list(coefficients.columns[-2:]))
    coefficients[f"[{alpha / 2:g}"] = [f"{v:.3f}" for v in lower_ci]
    coefficients[f"{1 - alpha / 2:g}]"] = [f"{v:.3f}" for v in upper_ci]

    # Create a bar plot with confidence intervals
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(coefficients["Feature"].values),
            y=coef,
            name="Estimated Coefficients",
            error_y=dict(
                type="data",
                symmetric=False,
                # error bars are distances from the coefficient, not bounds
                arrayminus=coef - lower_ci,
                array=upper_ci - coef,
                visible=True,
            ),
        )
    )

    fig.update_layout(
        title=f"{model.input_id} Coefficients with {confidence_level:.0%} Confidence Intervals",
        xaxis_title="Predictor Variables",
        yaxis_title="Coefficients",
    )

    return (
        fig,
        coefficients,
        RawData(
            model=model.input_id, std_err=std_err, lower_ci=lower_ci, upper_ci=upper_ci
        ),
    )
