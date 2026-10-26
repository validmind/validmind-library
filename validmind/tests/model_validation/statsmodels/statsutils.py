# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import re

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def adj_r2_score(
    actual: np.ndarray, predicted: np.ndarray, rowcount: int, featurecount: int
):
    """
    Adjusted R2 Score
    """
    return 1 - (1 - r2_score(actual, predicted)) * (rowcount - 1) / (
        rowcount - featurecount
    )


INTERCEPT_NAMES = ("Intercept", "const")


def ols_results(model, test_name: str):
    """
    Return the fitted statsmodels OLS results behind a VMModel.

    Works for models fitted with the formula API (`smf.ols`) and the array API
    (`sm.OLS`), on a DataFrame or on numpy arrays. Raises SkipTestError for any other
    model.
    """
    from validmind.errors import SkipTestError

    results = getattr(model, "model", None)
    fitted_model = getattr(results, "model", None)

    if getattr(model, "library", None) != "statsmodels" or fitted_model is None:
        raise SkipTestError(f"{test_name} requires a fitted statsmodels OLS model")

    from statsmodels.regression.linear_model import OLS

    if not isinstance(fitted_model, OLS):
        raise SkipTestError(
            f"{test_name} supports statsmodels OLS models, "
            f"not {type(fitted_model).__name__}"
        )

    return results


def row_labels(results):
    """Row labels of the data the model was fitted on (positions if none)."""
    labels = getattr(results.model.data, "row_labels", None)
    return list(labels) if labels is not None else list(range(int(results.nobs)))


def design_data(results, df, feature_columns=None):
    """
    Data to pass to `results.predict` / `results.get_prediction` for the rows of `df`.

    Formula models apply their own transformations to the original columns; array models
    need their regressor columns in order, with the constant added back. An array model
    fitted on numpy arrays names its regressors `x1, x2, ...`; those are taken to be
    `feature_columns`, in the same order.
    """
    from validmind.errors import SkipTestError

    if getattr(results.model, "formula", None) is not None:
        return df

    names = list(results.model.exog_names)
    regressors = [name for name in names if name not in INTERCEPT_NAMES]
    if all(name in df.columns for name in regressors):
        data = df.copy()
    elif feature_columns is not None and len(feature_columns) == len(regressors):
        data = pd.DataFrame(
            {
                name: df[column].to_numpy()
                for name, column in zip(regressors, feature_columns)
            },
            index=df.index,
        )
    else:
        raise SkipTestError(
            f"The model's regressors {regressors} are not dataset columns, and the "
            f"dataset has {len(feature_columns or [])} feature columns to match them "
            "by position. Fit the model on a DataFrame, or set the dataset's "
            "feature_columns in the order the model was fitted on."
        )
    for name in names:
        if name in INTERCEPT_NAMES:
            data[name] = 1.0
    return data[names]


def predict_frame(model, df, feature_columns):
    """
    Predictions of any regression VMModel for the rows of `df`.

    statsmodels models receive their design data (see `design_data`); every other model
    receives the dataset's feature columns as a DataFrame.
    """
    if getattr(model, "library", None) == "statsmodels":
        results = model.model
        return np.asarray(
            results.predict(design_data(results, df, feature_columns)), dtype=float
        )
    return np.asarray(model.predict(df[list(feature_columns)]), dtype=float)


def categorical_columns(model, columns):
    """
    The `columns` a statsmodels formula treats as categorical, such as `cards` in
    `C(cards)` or a string column. Empty for every other model.
    """
    results = getattr(model, "model", None)
    design_info = getattr(
        getattr(getattr(results, "model", None), "data", None), "design_info", None
    )
    if getattr(model, "library", None) != "statsmodels" or design_info is None:
        return set()

    codes = [
        factor.code
        for factor, info in design_info.factor_infos.items()
        if info.type == "categorical"
    ]
    return {
        column
        for column in columns
        if any(re.search(rf"\b{re.escape(column)}\b", code) for code in codes)
    }
