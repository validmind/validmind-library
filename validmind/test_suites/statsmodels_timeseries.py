# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Time Series Test Suites from statsmodels
"""

from validmind.vm_models import TestSuite


class RegressionModelDescription(TestSuite):
    """
    Test suite for performance metric of regression model of statsmodels library
    """

    suite_id = "regression_model_description"
    tests = [
        "validmind.data_validation.DatasetSplit",
        "validmind.model_validation.ModelMetadata",
    ]


class RegressionModelsEvaluation(TestSuite):
    """
    Test suite for metrics comparison of regression model of statsmodels library
    """

    suite_id = "regression_models_evaluation"
    tests = [
        "validmind.model_validation.statsmodels.RegressionModelCoeffs",
        "validmind.model_validation.sklearn.RegressionModelsPerformanceComparison",
    ]
