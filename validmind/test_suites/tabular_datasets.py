# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Test suites for tabular datasets
"""

from validmind.vm_models import TestSuite


class TabularDatasetDescription(TestSuite):
    """
    Test suite to extract metadata and descriptive
    statistics from a tabular dataset
    """

    suite_id = "tabular_dataset_description"
    tests = [
        "validmind.data_validation.DatasetDescription",
        "validmind.data_validation.DescriptiveStatistics",
        "validmind.data_validation.PearsonCorrelationMatrix",
    ]


class TabularDataQuality(TestSuite):
    """
    Test suite for data quality on tabular datasets
    """

    suite_id = "tabular_data_quality"
    tests = [
        "validmind.data_validation.ClassImbalance",
        "validmind.data_validation.Duplicates",
        "validmind.data_validation.HighCardinality",
        "validmind.data_validation.HighPearsonCorrelation",
        "validmind.data_validation.MissingValues",
        "validmind.data_validation.Skewness",
        "validmind.data_validation.UniqueRows",
        "validmind.data_validation.TooManyZeroValues",
    ]


class TabularDataset(TestSuite):
    """
    Test suite for tabular datasets.
    """

    suite_id = "tabular_dataset"
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
    ]
