# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for sklearn-compatible clustering models

Ideal setup is to have the API client to read a
custom test suite from the project's configuration
"""

from validmind.vm_models import TestSuite

from .parameters_optimization import KmeansParametersOptimization


class ClusterMetrics(TestSuite):
    """
    Test suite for sklearn clustering metrics
    """

    suite_id = "cluster_metrics"
    tests = [
        "validmind.model_validation.ModelMetadata",
        "validmind.data_validation.DatasetSplit",
        "validmind.model_validation.sklearn.HomogeneityScore",
        "validmind.model_validation.sklearn.CompletenessScore",
        "validmind.model_validation.sklearn.VMeasure",
        "validmind.model_validation.sklearn.AdjustedRandIndex",
        "validmind.model_validation.sklearn.AdjustedMutualInformation",
        "validmind.model_validation.sklearn.FowlkesMallowsScore",
        "validmind.model_validation.sklearn.ClusterPerformanceMetrics",
        "validmind.model_validation.sklearn.ClusterCosineSimilarity",
        "validmind.model_validation.sklearn.SilhouettePlot",
    ]


class ClusterPerformance(TestSuite):
    """
    Test suite for sklearn cluster performance
    """

    suite_id = "cluster_performance"
    tests = [
        "validmind.model_validation.ClusterSizeDistribution",
    ]


class ClusterFullSuite(TestSuite):
    """
    Full test suite for clustering models.
    """

    suite_id = "cluster_full_suite"
    tests = [
        {
            "section_id": ClusterMetrics.suite_id,
            "section_description": ClusterMetrics.__doc__,
            "section_tests": ClusterMetrics.tests,
        },
        {
            "section_id": ClusterPerformance.suite_id,
            "section_description": ClusterPerformance.__doc__,
            "section_tests": ClusterPerformance.tests,
        },
        {
            "section_id": KmeansParametersOptimization.suite_id,
            "section_description": KmeansParametersOptimization.__doc__,
            "section_tests": KmeansParametersOptimization.tests,
        },
    ]
