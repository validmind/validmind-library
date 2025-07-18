# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial


from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError, SkipTestError

try:
    from scipy import stats
except ImportError as e:
    if "scipy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scipy` for RegressionCoeffs. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["scipy"],
            extra="stats",
        ) from e

    raise e
from validmind.vm_models import VMModel


@tags("tabular_data", "visualization", "model_training")
@tasks("regression")
def RegressionCoeffs(model: VMModel) -> Tuple[go.Figure, RawData, pd.DataFrame]:
    """
    Assesses the significance and uncertainty of predictor variables in a regression model through visualization of
    coefficients and their 95% confidence intervals.

    ### Purpose

    The `RegressionCoeffs` metric visualizes the estimated regression coefficients alongside their 95% confidence intervals,
    providing insights into the impact and significance of predictor variables on the response variable. This visualization
    helps to understand the variability and uncertainty in the model's estimates, aiding in the evaluation of the
    significance of each predictor.

    ### Test Mechanism

    The function operates by extracting the estimated coefficients and their standard errors from the regression model.
    Using these, it calculates the confidence intervals at a 95% confidence level, which indicates the range within which
    the true coefficient value is expected to fall 95% of the time. The confidence intervals are computed using the
    Z-value associated with the 95% confidence level. The coefficients and their confidence intervals are then visualized
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

    # Extract estimated coefficients and standard errors
    coefficients = model.regression_coefficients()
    coef = pd.to_numeric(coefficients["coef"])
    std_err = pd.to_numeric(coefficients["std err"])

    # Calculate confidence intervals
    confidence_level = 0.95  # 95% confidence interval
    z_value = stats.norm.ppf((1 + confidence_level) / 2)  # Calculate Z-value
    lower_ci = coef - z_value * std_err
    upper_ci = coef + z_value * std_err

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
                arrayminus=lower_ci,
                array=upper_ci,
                visible=True,
            ),
        )
    )

    fig.update_layout(
        title=f"{model.input_id} Coefficients with Confidence Intervals",
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
