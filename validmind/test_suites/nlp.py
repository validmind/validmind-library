# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for NLP models
"""

from validmind.vm_models import TestSuite

from .classifier import ClassifierDiagnosis, ClassifierMetrics, ClassifierPerformance
from .text_data import TextDataQuality


class NLPClassifierFullSuite(TestSuite):
    """
    Full test suite for NLP classification models.
    """

    suite_id = "nlp_classifier_full_suite"
    tests = [
        {
            "section_id": TextDataQuality.suite_id,
            "section_description": TextDataQuality.__doc__,
            "section_tests": TextDataQuality.tests,
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
