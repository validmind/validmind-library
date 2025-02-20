# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import numpy as np
import pandas as pd

from validmind.errors import MissingRExtrasError
from validmind.logging import get_logger
from validmind.vm_models.model import VMModel

logger = get_logger(__name__)


def get_full_class_name(obj):
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


class RModel(VMModel):
    """An R model class that wraps a "fitted" R model instance and its associated data."""

    def __post_init__(self):
        self.language = self.r["version"].rx2("version.string")[0]
        self.library = self.class_ = "R"

        name_map = {
            "xgb.Booster": "XGBoost",
            "glm": self.__glm_model_class(),
            "lm": "Linear Regression",
        }
        self.name = self.name or name_map.get(
            self.__model_class(), self.__model_class()
        )

        self._is_classification_model = self.__is_classification_model()

    def __is_classification_model(self):
        """
        Only supported classification models are XGBClassifier and GLM with binomial family (logistic regression).

        Since R uses a single global predict() method for any model, we try to figure out
        if the predict() method called by our tests is supposed to return probabilities or classes
        """
        model_class_name = self.__model_class()

        if model_class_name == "xgb.Booster":
            model_params = self.model.rx2("params")
            model_objective = model_params.rx2("objective")[0]
            return model_objective == "binary:logistic"
        elif model_class_name == "glm":
            model_family = self.model.rx2("family").rx2("family")[0]
            return model_family == "binomial"

        return False

    def __get_predict_data_as_df(self, new_data):
        """
        Builds the correct data shape and format for the predict method when the
        caller has passed a Pandas dataframe as input. This function makes sure to
        adjust the shape of the input dataset to the predict() signature depending
        if it's a regular R model or an XGBoost model
        """
        if self.__model_class() == "xgb.Booster":
            return new_data.df.drop(new_data.target_column, axis=1)

        return new_data.df

    def __model_class(self):
        """
        Returns the model class name
        """
        return self.r["class"](self.model)[0]

    def __glm_model_class(self):
        """
        Returns the model class name for GLM models which include family and link function
        """
        # Access the attributes of the model
        model_family = self.model.rx2("family")
        model_method = self.model.rx2("method")[0]

        # Extract the family name and link function
        family_name = model_family.rx2("family")[0]
        link_function = model_family.rx2("link")[0]

        # Reconstruct the string in Python
        output = (
            f"{model_method} (Family: {family_name}, Link function: {link_function})"
        )

        return output

    def r_predict(self, new_data_r):
        """
        Predict method for the model. This is a wrapper around R's global predict.

        An R model doesn't provide separate predict() and predict_proba() methods.
        Instead, there is a global predict() method that returns the predicted
        values according to the model type.
        """
        # Use the predict method on the loaded model (assuming the model's name in R is 'model')
        predicted_probs = self.r.predict(
            self.model, newdata=new_data_r, type="response"
        )
        return predicted_probs

    def r_xgb_predict(self, new_data_r):
        """
        Predict method for XGBoost models. This is a wrapper around R's global predict
        """
        new_data_matrix = self.r["as.matrix"](new_data_r)
        new_data_r = self.r["xgb.DMatrix"](new_data_matrix)

        predicted_probs = self.r.predict(
            self.model, newdata=new_data_r, type="response"
        )
        return predicted_probs

    def predict_proba(self, new_data):
        """
        Calls the R global predict method with type="response" to get the predicted probabilities
        """
        return self.predict(new_data, return_probs=True)

    def predict(self, new_data, return_probs=False):
        """
        Converts the predicted probabilities to classes
        """
        try:
            from rpy2.robjects import pandas2ri
        except ImportError:
            raise MissingRExtrasError()

        # Activate the pandas conversion for rpy2
        pandas2ri.activate()

        new_data_class = get_full_class_name(new_data)

        if new_data_class == "numpy.ndarray":
            # We need to reconstruct the DataFrame from the ndarray using the column names
            new_data = pd.DataFrame(new_data, columns=self.test_ds.feature_columns)
        elif new_data_class != "pandas.core.frame.DataFrame":
            raise ValueError(
                f"new_data must be a DataFrame or ndarray. Got {new_data_class}"
            )

        new_data_r = pandas2ri.py2rpy(new_data)

        if self.__model_class() == "xgb.Booster":
            predicted_probs = self.r_xgb_predict(new_data_r)
        else:
            predicted_probs = self.r_predict(new_data_r)

        if self._is_classification_model and return_probs is False:
            predicted_classes = np.where(predicted_probs > 0.5, 1, 0)
            return predicted_classes

        return predicted_probs

    def regression_coefficients(self):
        """
        Returns the regression coefficients summary of the model
        """
        # Each list in this list contains the coefficients for one feature, starting with the intercept
        coefficient_values = self.r.summary(self.model).rx2("coefficients")

        # Extract the coefficient names by calling the model terms.
        # [[1]] is the endogenous variable and the rest are the coefficients
        model_terms = self.model.rx2["terms"]
        variables_attr = self.r.attr(model_terms, "variables")

        variables_list = [str(variables_attr[i]) for i in range(1, len(variables_attr))]
        # Remove "Const" from the list of variables for Poisson regression
        variables_list = [v for v in variables_list if v != "Const"]

        # Build a dataframe where each row is a feature (including the intercept) and each column is one
        # of the values in the coefficient_values list, which has: [Estimate, Std. Error, t value, Pr(>|t|)]
        # We use ["coef", "std errr", "z", "P>|z|"] to stay consistent with the statsmodels implementation
        # Remove the first row which is the intercept
        coefficient_values = coefficient_values[1:]
        coefficients_df = pd.DataFrame(
            coefficient_values,
            columns=["coef", "std err", "z", "P>|z|"],
        )

        # Add the feature names as a column and rearrange to have feature name as the first column
        coefficients_df["Feature"] = variables_list[1:]
        coefficients_df = coefficients_df[["Feature"] + list(coefficients_df.columns)]

        return coefficients_df
