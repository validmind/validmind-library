# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for llm summarization models
"""

from validmind.vm_models import TestSuite


class SummarizationMetrics(TestSuite):
    """
    Test suite for Summarization metrics
    """

    suite_id = "summarization_metrics"
    tests = [
        "validmind.model_validation.TokenDisparity",
        "validmind.model_validation.BleuScore",
        "validmind.model_validation.BertScore",
        "validmind.model_validation.ContextualRecall",
    ]
