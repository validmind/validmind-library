# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from validmind import RawData, tags, tasks
from validmind.errors import MissingDependencyError
from validmind.vm_models import VMModel

from .statsutils import INTERCEPT_NAMES, ols_results

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError as e:
    if "statsmodels" in str(e):
        raise MissingDependencyError(
            "Missing required package `statsmodels` for VarianceInflationFactor. "
            "Please run `pip install validmind[stats]` to use statistical tests",
            required_dependencies=["statsmodels"],
            extra="stats",
        ) from e

    raise e


@tags("model_diagnosis", "multicollinearity", "visualization", "statsmodels")
@tasks("regression")
def VarianceInflationFactor(
    model: VMModel,
    vif_threshold: float = 10.0,
    moderate_vif_threshold: float = 5.0,
    condition_number_threshold: float = 30.0,
) -> Tuple[Dict[str, pd.DataFrame], go.Figure, bool, RawData]:
    """
    Detects multicollinearity among the regressors of a linear regression model using the
    variance inflation factor (VIF) and the condition number of the design matrix.

    ### Purpose

    Multicollinearity — regressors that carry nearly the same information — inflates the
    standard errors of the coefficients, makes their signs and sizes unstable, and makes the
    contribution of each regressor impossible to separate. The model can still predict well,
    which is why this needs a dedicated check: it is invisible in fit metrics such as R².

    ### Test Mechanism

    For each regressor j (excluding the intercept), the VIF is `1 / (1 - R²_j)`, where `R²_j`
    is the R² of regressing that column on all other columns of the design matrix. A VIF of 5
    means the variance of the coefficient is five times what it would be with uncorrelated
    regressors. Each regressor is labelled OK, Moderate (`>= moderate_vif_threshold`) or
    Severe (`>= vif_threshold`).

    The condition number is computed on the design matrix with every column scaled to unit
    length (Belsley's scaling), so it does not depend on the units of the regressors; values
    above `condition_number_threshold` (30 by convention) indicate harmful collinearity.

    The test passes when no regressor reaches `vif_threshold` and the scaled condition number
    is below `condition_number_threshold`.

    ### Signs of High Risk

    - One or more regressors with a VIF at or above 10.
    - Pairs of regressors with similarly high VIFs, pointing to two variables that measure the
    same thing.
    - A scaled condition number above 30.

    ### Strengths

    - Quantifies multicollinearity per regressor, so the variables involved are identified.
    - Independent of the units of the regressors (condition number is scaled).
    - Works on the model's own design matrix, including transformed and dummy-coded terms.

    ### Limitations

    - Requires a statsmodels OLS model.
    - Dummy columns of one categorical variable are assessed separately; a generalized VIF
    would assess them together.
    - VIF identifies collinearity but not which of the collinear variables to keep — that is a
    modelling judgement.
    """
    results = ols_results(model, "VarianceInflationFactor")
    exog = np.asarray(results.model.exog, dtype=float)
    names = list(results.model.exog_names)

    rows = []
    for i, name in enumerate(names):
        if name in INTERCEPT_NAMES:
            continue
        vif = float(variance_inflation_factor(exog, i))
        if vif >= vif_threshold:
            status = "Severe"
        elif vif >= moderate_vif_threshold:
            status = "Moderate"
        else:
            status = "OK"
        rows.append({"Feature": name, "VIF": round(vif, 2), "Status": status})

    vif_df = pd.DataFrame(rows)

    scaled = exog / np.linalg.norm(exog, axis=0)
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    condition_number = float(singular_values.max() / singular_values.min())

    condition_df = pd.DataFrame(
        [
            {
                "Scaled Condition Number": round(condition_number, 2),
                "Threshold": condition_number_threshold,
                "Pass/Fail": (
                    "Pass" if condition_number < condition_number_threshold else "Fail"
                ),
            }
        ]
    )

    colors = {"OK": "#2e7d32", "Moderate": "#f9a825", "Severe": "#c62828"}
    fig = go.Figure(
        go.Bar(
            x=vif_df["Feature"],
            y=vif_df["VIF"],
            marker_color=[colors[s] for s in vif_df["Status"]],
        )
    )
    for threshold, label in [
        (moderate_vif_threshold, "Moderate"),
        (vif_threshold, "Severe"),
    ]:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text=f"{label} ({threshold})",
            annotation_position="top left",
        )
    fig.update_layout(
        title=f"Variance Inflation Factor — {model.input_id}",
        xaxis_title="Regressor",
        yaxis_title="VIF",
    )

    passed = bool(
        (vif_df["VIF"] < vif_threshold).all()
        and condition_number < condition_number_threshold
    )

    return (
        {"Variance Inflation Factor": vif_df, "Condition Number": condition_df},
        fig,
        passed,
        RawData(
            vif=vif_df,
            condition_number=condition_number,
            model=model.input_id,
        ),
    )
