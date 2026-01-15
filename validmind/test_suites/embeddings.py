# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for embeddings models

Ideal setup is to have the API client to read a
custom test suite from the project's configuration
"""

from validmind.vm_models import TestSuite


class EmbeddingsMetrics(TestSuite):
    """
    Test suite for embeddings metrics
    """

    suite_id = "embeddings_metrics"
    tests = [
        "validmind.model_validation.ModelMetadata",
        "validmind.data_validation.DatasetSplit",
        "validmind.model_validation.embeddings.DescriptiveAnalytics",
        "validmind.model_validation.embeddings.CosineSimilarityDistribution",
        "validmind.model_validation.embeddings.ClusterDistribution",
        "validmind.model_validation.embeddings.EmbeddingsVisualization2D",
    ]


class EmbeddingsPerformance(TestSuite):
    """
    Test suite for embeddings model performance
    """

    suite_id = "embeddings_model_performance"
    tests = [
        "validmind.model_validation.embeddings.StabilityAnalysisRandomNoise",
        "validmind.model_validation.embeddings.StabilityAnalysisSynonyms",
        "validmind.model_validation.embeddings.StabilityAnalysisKeyword",
        "validmind.model_validation.embeddings.StabilityAnalysisTranslation",
    ]


class EmbeddingsFullSuite(TestSuite):
    """
    Full test suite for embeddings models.
    """

    suite_id = "embeddings_full_suite"
    tests = [
        {
            "section_id": EmbeddingsMetrics.suite_id,
            "section_description": EmbeddingsMetrics.__doc__,
            "section_tests": EmbeddingsMetrics.tests,
        },
        {
            "section_id": EmbeddingsPerformance.suite_id,
            "section_description": EmbeddingsPerformance.__doc__,
            "section_tests": EmbeddingsPerformance.tests,
        },
    ]
