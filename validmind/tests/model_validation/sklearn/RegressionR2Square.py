# Copyright © 2023 ValidMind Inc. All rights reserved.

from dataclasses import dataclass

from sklearn import metrics

from validmind.tests.model_validation.statsmodels.statsutils import adj_r2_score
from validmind.vm_models import Metric, ResultSummary, ResultTable


@dataclass
class RegressionR2Square(Metric):
    """
    **Purpose**: The purpose of the RegressionR2Square Metric test is to measure the overall goodness-of-fit of a
    regression model. Specifically, this Python-based test evaluates the R-squared (R2) and Adjusted R-squared (Adj R2)
    scores: two statistical measures within regression analysis used to evaluate the strength of the relationship
    between the model's predictors and the response variable.

    **Test Mechanism**: The test deploys the 'r2_score' method from the Scikit-learn metrics module, measuring the R2
    score on both training and test sets. This score reflects the proportion of the variance in the dependent variable
    that is predictable from independent variables. The test also considers the Adjusted R2 score, accounting for the
    number of predictors in the model, to penalize model complexity and thus reduce overfitting. The Adjusted R2 score
    will be smaller if unnecessary predictors are included in the model.

    **Signs of High Risk**: Indicators of high risk in this test may include a low R2 or Adjusted R2 score, which would
    suggest that the model does not explain much variation in the dependent variable. The occurrence of overfitting is
    also a high-risk sign, evident when the R2 score on the training set is significantly higher than on the test set,
    indicating that the model is not generalizing well to unseen data.

    **Strengths**: The R2 score is a widely-used measure in regression analysis, providing a sound general indication
    of model performance. It is easy to interpret and understand, as it is essentially representing the proportion of
    the dependent variable's variance explained by the independent variables. The Adjusted R2 score complements the R2
    score well by taking into account the number of predictors in the model, which helps control overfitting.

    **Limitations**: R2 and Adjusted R2 scores can be sensitive to the inclusion of unnecessary predictors in the model
    (even though Adjusted R2 is intended to penalize complexity). Their reliability might also lessen in cases of
    non-linear relationships or when the underlying assumptions of linear regression are violated. Additionally, while
    they summarize how well the model fits the data, they do not provide insight on whether the correct regression was
    used, or whether certain key assumptions have been fulfilled.
    """

    category = "model_performance"
    name = "regression_errors"
    required_inputs = ["model"]
    metadata = {
        "task_types": ["regression"],
        "tags": [
            "sklearn",
            "model_performance",
        ],
    }

    def summary(self, raw_results):

        """
        Returns a summarized representation of the dataset split information
        """
        table_records = []
        for result in raw_results:
            for key, value in result.items():
                table_records.append(
                    {
                        "Metric": key,
                        "TRAIN": result[key]["train"],
                        "TEST": result[key]["test"],
                    }
                )

        return ResultSummary(results=[ResultTable(data=table_records)])

    def run(self):
        y_true_train = self.model.y_train_true
        class_pred_train = self.model.y_train_predict
        y_true_train = y_true_train.astype(class_pred_train.dtype)

        y_true_test = self.model.y_test_true
        class_pred_test = self.model.y_test_predict
        y_true_test = y_true_test.astype(class_pred_test.dtype)

        r2s_train = metrics.r2_score(y_true_train, class_pred_train)
        r2s_test = metrics.r2_score(y_true_test, class_pred_test)

        results = []
        results.append(
            {
                "R-squared (R2) Score": {
                    "train": r2s_train,
                    "test": r2s_test,
                }
            }
        )

        X_columns = self.model.train_ds.get_features_columns()
        adj_r2_train = adj_r2_score(
            y_true_train, class_pred_train, len(y_true_train), len(X_columns)
        )
        adj_r2_test = adj_r2_score(
            y_true_test, class_pred_test, len(y_true_test), len(X_columns)
        )
        results.append(
            {
                "Adjusted R-squared (R2) Score": {
                    "train": adj_r2_train,
                    "test": adj_r2_test,
                }
            }
        )
        return self.cache_results(metric_value=results)