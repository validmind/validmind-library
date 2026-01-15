# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for text datasets
"""

from validmind.vm_models import TestSuite


class TextDataQuality(TestSuite):
    """
    Test suite for data quality on text data
    """

    suite_id = "text_data_quality"
    tests = [
        "validmind.data_validation.ClassImbalance",
        "validmind.data_validation.Duplicates",
        "validmind.data_validation.nlp.StopWords",
        "validmind.data_validation.nlp.Punctuations",
        "validmind.data_validation.nlp.CommonWords",
        "validmind.data_validation.nlp.TextDescription",
    ]
