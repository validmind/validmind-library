# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for sklearn-compatible classifier models

Ideal setup is to have the API client to read a
custom test suite from the project's configuration
"""

from validmind.vm_models import TestSuite

from .tabular_datasets import TabularDataQuality, TabularDatasetDescription


class ClassifierMetrics(TestSuite):
    """
    Test suite for sklearn classifier metrics
    """

    suite_id = "classifier_metrics"
    tests = [
        "validmind.model_validation.ModelMetadata",
        "validmind.data_validation.DatasetSplit",
        "validmind.model_validation.sklearn.ConfusionMatrix",
        "validmind.model_validation.sklearn.ClassifierPerformance",
        "validmind.model_validation.sklearn.PermutationFeatureImportance",
        "validmind.model_validation.sklearn.PrecisionRecallCurve",
        "validmind.model_validation.sklearn.ROCCurve",
        "validmind.model_validation.sklearn.PopulationStabilityIndex",
        "validmind.model_validation.sklearn.SHAPGlobalImportance",
    ]


class ClassifierPerformance(TestSuite):
    """
    Test suite for sklearn classifier models
    """

    suite_id = "classifier_validation"
    tests = [
        "validmind.model_validation.sklearn.MinimumAccuracy",
        "validmind.model_validation.sklearn.MinimumF1Score",
        "validmind.model_validation.sklearn.MinimumROCAUCScore",
        "validmind.model_validation.sklearn.TrainingTestDegradation",
        "validmind.model_validation.sklearn.ModelsPerformanceComparison",
    ]


class ClassifierDiagnosis(TestSuite):
    """
    Test suite for sklearn classifier model diagnosis tests
    """

    suite_id = "classifier_model_diagnosis"
    tests = [
        "validmind.model_validation.sklearn.OverfitDiagnosis",
        "validmind.model_validation.sklearn.WeakspotsDiagnosis",
        "validmind.model_validation.sklearn.RobustnessDiagnosis",
    ]


class ClassifierModelValidation(TestSuite):
    """
    Test suite for binary classification models.
    """

    suite_id = "classifier_model_validation"
    tests = [
        {
            "section_id": ClassifierMetrics.suite_id,
            "section_description": ClassifierMetrics.__doc__,
            "section_tests": ClassifierMetrics.tests,
        },
        {
            "section_id": ClassifierPerformance.suite_id,
            "section_description": ClassifierPerformance.__doc__,
            "section_tests": ClassifierPerformance.tests,
        },
        {
            "section_id": ClassifierDiagnosis.suite_id,
            "section_description": ClassifierDiagnosis.__doc__,
            "section_tests": ClassifierDiagnosis.tests,
        },
    ]


class ClassifierFullSuite(TestSuite):
    """
    Full test suite for binary classification models.
    """

    suite_id = "classifier_full_suite"
    tests = [
        {
            "section_id": TabularDatasetDescription.suite_id,
            "section_description": TabularDatasetDescription.__doc__,
            "section_tests": TabularDatasetDescription.tests,
        },
        {
            "section_id": TabularDataQuality.suite_id,
            "section_description": TabularDataQuality.__doc__,
            "section_tests": TabularDataQuality.tests,
        },
        {
            "section_id": ClassifierMetrics.suite_id,
            "section_description": ClassifierMetrics.__doc__,
            "section_tests": ClassifierMetrics.tests,
        },
        {
            "section_id": ClassifierPerformance.suite_id,
            "section_description": ClassifierPerformance.__doc__,
            "section_tests": ClassifierPerformance.tests,
        },
        {
            "section_id": ClassifierDiagnosis.suite_id,
            "section_description": ClassifierDiagnosis.__doc__,
            "section_tests": ClassifierDiagnosis.tests,
        },
    ]
