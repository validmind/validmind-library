# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for sklearn-compatible hyper parameters tunning

Ideal setup is to have the API client to read a
custom test suite from the project's configuration
"""

from validmind.vm_models import TestSuite


class KmeansParametersOptimization(TestSuite):
    """
    Test suite for sklearn hyperparameters optimization
    """

    suite_id = "hyper_parameters_optimization"
    tests = [
        "validmind.model_validation.sklearn.HyperParametersTuning",
        "validmind.model_validation.sklearn.KMeansClustersOptimization",
    ]
