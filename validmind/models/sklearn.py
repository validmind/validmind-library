# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import pandas as pd

from validmind.errors import MissingOrInvalidModelPredictFnError
from validmind.logging import get_logger
from validmind.vm_models.model import VMModel, has_method_with_arguments

logger = get_logger(__name__)


class SKlearnModel(VMModel):
    def __post_init__(self):
        if not self.model:
            raise ValueError("Model object is a required argument for SKlearnModel")

        self.library = self.model.__class__.__module__.split(".")[0]
        self.class_ = self.model.__class__.__name__
        self.name = self.name or type(self.model).__name__

    def predict_proba(self, *args, **kwargs):
        """
        predict_proba (for classification) or predict (for regression) method
        """
        if not has_method_with_arguments(self.model, "predict_proba", 1):
            raise MissingOrInvalidModelPredictFnError(
                f"SKlearn model {self.model.__class__} Model does not have a compatible predict_proba implementation."
                + " Please assign predictions directly with vm_dataset.assign_predictions(model, prediction_values)"
            )
        if callable(getattr(self.model, "predict_proba", None)):
            return self.model.predict_proba(*args, **kwargs)[:, 1]
        return None

    def predict(self, *args, **kwargs):
        """
        Predict method for the model. This is a wrapper around the model's
        """
        return self.model.predict(*args, **kwargs)


class CatBoostModel(SKlearnModel):
    """Wrapper for CatBoost model"""

    pass


class XGBoostModel(SKlearnModel):
    """Wrapper for XGBoost model"""

    def __post_init__(self):
        super().__post_init__()
        self.library = "xgboost"


class StatsModelsModel(SKlearnModel):
    """Wrapper for StatsModels model"""

    def __post_init__(self):
        super().__post_init__()
        self.library = "statsmodels"

    def regression_coefficients(self):
        """
        Returns the regression coefficients summary of the model
        """
        raw_summary = self.model.summary()

        table = raw_summary.tables[1].data
        headers = table.pop(0)
        headers[0] = "Feature"

        return pd.DataFrame(table, columns=headers)
