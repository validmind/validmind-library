# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import numpy as np
import pandas as pd

from validmind import RawData, tags, tasks
from validmind.vm_models import VMModel

from .statsutils import ols_results


@tags("model_diagnosis", "statistical_test", "statsmodels")
@tasks("regression")
def RobustStandardErrors(
    model: VMModel,
    cov_type: str = "HC3",
    significance_level: float = 0.05,
) -> Tuple[pd.DataFrame, bool, RawData]:
    """
    Compares a linear regression model's ordinary standard errors and p-values with
    heteroscedasticity-robust ones, and flags coefficients whose significance depends on
    which is used.

    ### Purpose

    Ordinary least squares standard errors assume the error variance is the same for every
    observation. When it is not (heteroscedasticity), they are wrong — usually too small —
    and so are the p-values and confidence intervals built on them. Robust (sandwich)
    standard errors remain valid without that assumption. If a coefficient is significant
    only under the ordinary standard errors, the evidence for it is weaker than the model
    summary suggests.

    ### Test Mechanism

    The model's covariance matrix is re-estimated with the robust estimator `cov_type`
    (default `HC3`, which performs well in finite samples; `HC0`-`HC2` are also accepted).
    For each coefficient the test reports both standard errors, their ratio, both p-values,
    and whether the coefficient is significant at `significance_level` under each.

    The test passes when no coefficient changes significance.

    ### Signs of High Risk

    - A coefficient significant under ordinary standard errors but not under robust ones.
    - Robust standard errors much larger than ordinary ones (ratio well above 1).

    ### Strengths

    - Directly measures how much the inference depends on the constant-variance assumption.
    - Leaves the coefficients unchanged — only their uncertainty is re-assessed.

    ### Limitations

    - Requires a statsmodels OLS model.
    - Robust standard errors correct inference, not bias: a misspecified model stays
    misspecified.
    - Does not account for clustered or autocorrelated errors.
    """
    # refit: a model fitted with .fit(cov_type=...) already carries robust errors in bse
    results = ols_results(model, "RobustStandardErrors").model.fit()
    robust = results.get_robustcov_results(cov_type=cov_type)

    ols_se = np.asarray(results.bse, dtype=float)
    robust_se = np.asarray(robust.bse, dtype=float)
    ols_p = np.asarray(results.pvalues, dtype=float)
    robust_p = np.asarray(robust.pvalues, dtype=float)

    table = pd.DataFrame(
        {
            "Coefficient": results.model.exog_names,
            "Estimate": np.asarray(results.params, dtype=float),
            "OLS Std. Error": ols_se,
            f"{cov_type} Std. Error": robust_se,
            "Ratio (Robust / OLS)": robust_se / ols_se,
            "OLS p-value": ols_p,
            f"{cov_type} p-value": robust_p,
            "Significant (OLS)": ols_p < significance_level,
            f"Significant ({cov_type})": robust_p < significance_level,
        }
    )
    table["Significance Changes"] = (
        table["Significant (OLS)"] != table[f"Significant ({cov_type})"]
    )

    passed = bool(not table["Significance Changes"].any())

    return (
        table.round(6),
        passed,
        RawData(
            ols_se=ols_se,
            robust_se=robust_se,
            cov_type=cov_type,
            model=model.input_id,
        ),
    )
