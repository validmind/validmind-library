# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Tuple

import pandas as pd
from sklearn import metrics

from validmind import RawData, tags, tasks
from validmind.tests.model_validation.statsmodels.statsutils import adj_r2_score
from validmind.vm_models import VMDataset, VMModel


@tags("sklearn", "model_performance")
@tasks("regression")
def RegressionR2Square(
    dataset: VMDataset, model: VMModel
) -> Tuple[pd.DataFrame, RawData]:
    """
    Assesses the overall goodness-of-fit of a regression model by evaluating R-squared (R2) and Adjusted R-squared (Adj
    R2) scores to determine the model's explanatory power over the dependent variable.

    ### Purpose

    The purpose of the RegressionR2Square Metric test is to measure the overall goodness-of-fit of a regression model.
    Specifically, this Python-based test evaluates the R-squared (R2) and Adjusted R-squared (Adj R2) scores, which are
    statistical measures used to assess the strength of the relationship between the model's predictors and the
    response variable.

    ### Test Mechanism

    The test deploys the `r2_score` method from the Scikit-learn metrics module to measure the R2 score on both
    training and test sets. This score reflects the proportion of the variance in the dependent variable that is
    predictable from the independent variables. The test also calculates the Adjusted R2 score, which accounts for the
    number of predictors in the model to penalize model complexity and reduce overfitting. The Adjusted R2 score will
    be smaller if unnecessary predictors are included in the model.

    ### Signs of High Risk

    - Low R2 or Adjusted R2 scores, suggesting that the model does not explain much variation in the dependent variable.
    - Significant discrepancy between R2 scores on the training set and test set, indicating overfitting and poor
    generalization to unseen data.

    ### Strengths

    - Widely-used measure in regression analysis, providing a sound general indication of model performance.
    - Easy to interpret and understand, as it represents the proportion of the dependent variable's variance explained
    by the independent variables.
    - Adjusted R2 score helps control overfitting by penalizing unnecessary predictors.

    ### Limitations

    - Sensitive to the inclusion of unnecessary predictors even though Adjusted R2 penalizes complexity.
    - Less reliable in cases of non-linear relationships or when the underlying assumptions of linear regression are
    violated.
    - Does not provide insight on whether the correct regression model was used or if key assumptions have been met.
    """
    y_true = dataset.y
    y_pred = dataset.y_pred(model)
    y_true = y_true.astype(y_pred.dtype)

    r2 = metrics.r2_score(y_true, y_pred)
    adj_r2 = adj_r2_score(y_true, y_pred, len(y_true), len(dataset.feature_columns))

    return pd.DataFrame(
        {
            "R-squared (R2) Score": [r2],
            "Adjusted R-squared (R2) Score": [adj_r2],
        }
    ), RawData(
        y_true=y_true, y_pred=y_pred, model=model.input_id, dataset=dataset.input_id
    )
