# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Dict, Tuple, Union

import pandas as pd

from validmind import RawData, tags, tasks
from validmind.vm_models import VMModel

from .statsutils import ols_results

_SIGNS = {"+": 1, "positive": 1, 1: 1, "-": -1, "negative": -1, -1: -1}


@tags("model_diagnosis", "conceptual_soundness", "statsmodels")
@tasks("regression")
def ExpectedCoefficientSigns(
    model: VMModel,
    expected_signs: Dict[str, Union[str, int]],
    significance_level: float = 0.05,
) -> Tuple[pd.DataFrame, bool, RawData]:
    """
    Checks that each coefficient of a linear regression model has the sign that business
    knowledge or economic theory expects.

    ### Purpose

    Conceptual soundness requires that a model's effects make sense: higher income should
    not raise predicted default, a higher price should not raise predicted demand. A
    coefficient with the wrong sign usually points to multicollinearity, an omitted
    variable, or a data problem — and it undermines the model's use for any decision that
    relies on its explanations, even when its predictions are accurate.

    ### Test Mechanism

    The validator supplies `expected_signs`, mapping coefficient names (as they appear in
    the model, for example `"Income"` or `"C(Region)[T.West]"`) to `"+"` / `"-"` (or
    `"positive"` / `"negative"`, `1` / `-1`). For each listed coefficient the test reports
    the estimate, its p-value and its actual sign:

    - **Pass:** the sign matches.
    - **Fail:** the sign is opposite and the coefficient is significant at
    `significance_level` — the data contradicts the expectation.
    - **Fail (not significant):** the sign is opposite but the coefficient is not
    significant — the data does not support the expected effect either.

    Coefficients not listed are not assessed. The test passes when every listed coefficient
    passes.

    ### Signs of High Risk

    - A significant coefficient with the opposite sign to the expectation.
    - Several sign failures among collinear regressors.

    ### Strengths

    - Makes the expected direction of each effect explicit and auditable.
    - Distinguishes a contradicted expectation from an unsupported one.

    ### Limitations

    - Requires a statsmodels OLS model.
    - Only as good as the expectations supplied; it does not judge magnitudes.
    """
    if not expected_signs:
        raise ValueError("expected_signs must name at least one coefficient")
    results = ols_results(model, "ExpectedCoefficientSigns")
    # a model fitted on numpy arrays has array params, named x1, x2, ... by statsmodels
    names = results.model.exog_names
    params = pd.Series(results.params, index=names)
    pvalues = pd.Series(results.pvalues, index=names)

    unknown = [name for name in expected_signs if name not in params.index]
    if unknown:
        raise ValueError(
            f"expected_signs names not in the model: {unknown}. "
            f"Model coefficients: {list(params.index)}"
        )

    rows = []
    for name, expected in expected_signs.items():
        key = expected.lower() if isinstance(expected, str) else expected
        if key not in _SIGNS:
            raise ValueError(
                f"Expected sign for {name!r} must be '+', '-', 'positive', "
                f"'negative', 1 or -1, got {expected!r}"
            )
        estimate = float(params[name])
        p_value = float(pvalues[name])
        matches = (estimate > 0) == (_SIGNS[key] > 0)
        if matches:
            result = "Pass"
        elif p_value < significance_level:
            result = "Fail"
        else:
            result = "Fail (not significant)"
        rows.append(
            {
                "Coefficient": name,
                "Expected Sign": "+" if _SIGNS[key] > 0 else "-",
                "Estimate": round(estimate, 6),
                "Actual Sign": "+" if estimate > 0 else "-",
                "p-value": round(p_value, 6),
                "Result": result,
            }
        )

    table = pd.DataFrame(rows)

    return (
        table,
        bool((table["Result"] == "Pass").all()),
        RawData(expected_signs=dict(expected_signs), model=model.input_id),
    )
