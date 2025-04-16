# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Entrypoint for test suites.
"""
from inspect import getdoc

import pandas as pd

from ..logging import get_logger
from ..tests import load_test
from ..utils import format_dataframe, test_id_to_name
from ..vm_models import TestSuite
from .classifier import (
    ClassifierDiagnosis,
    ClassifierFullSuite,
    ClassifierMetrics,
    ClassifierModelValidation,
    ClassifierPerformance,
)
from .cluster import ClusterFullSuite, ClusterMetrics, ClusterPerformance
from .embeddings import EmbeddingsFullSuite, EmbeddingsMetrics, EmbeddingsPerformance
from .llm import LLMClassifierFullSuite, PromptValidation
from .nlp import NLPClassifierFullSuite
from .parameters_optimization import KmeansParametersOptimization
from .regression import RegressionFullSuite, RegressionMetrics, RegressionPerformance
from .statsmodels_timeseries import (
    RegressionModelDescription,
    RegressionModelsEvaluation,
)
from .summarization import SummarizationMetrics
from .tabular_datasets import (
    TabularDataQuality,
    TabularDataset,
    TabularDatasetDescription,
)
from .text_data import TextDataQuality
from .time_series import (
    TimeSeriesDataQuality,
    TimeSeriesDataset,
    TimeSeriesModelValidation,
    TimeSeriesMultivariate,
    TimeSeriesUnivariate,
)

logger = get_logger(__name__)

core_test_suites = {
    ClassifierDiagnosis.suite_id: ClassifierDiagnosis,
    ClassifierFullSuite.suite_id: ClassifierFullSuite,
    ClassifierMetrics.suite_id: ClassifierMetrics,
    ClassifierModelValidation.suite_id: ClassifierModelValidation,
    ClassifierPerformance.suite_id: ClassifierPerformance,
    ClusterFullSuite.suite_id: ClusterFullSuite,
    ClusterMetrics.suite_id: ClusterMetrics,
    ClusterPerformance.suite_id: ClusterPerformance,
    EmbeddingsFullSuite.suite_id: EmbeddingsFullSuite,
    EmbeddingsMetrics.suite_id: EmbeddingsMetrics,
    EmbeddingsPerformance.suite_id: EmbeddingsPerformance,
    KmeansParametersOptimization.suite_id: KmeansParametersOptimization,
    LLMClassifierFullSuite.suite_id: LLMClassifierFullSuite,
    PromptValidation.suite_id: PromptValidation,
    NLPClassifierFullSuite.suite_id: NLPClassifierFullSuite,
    RegressionMetrics.suite_id: RegressionMetrics,
    RegressionModelDescription.suite_id: RegressionModelDescription,
    RegressionModelsEvaluation.suite_id: RegressionModelsEvaluation,
    RegressionFullSuite.suite_id: RegressionFullSuite,
    RegressionPerformance.suite_id: RegressionPerformance,
    SummarizationMetrics.suite_id: SummarizationMetrics,
    TabularDataset.suite_id: TabularDataset,
    TabularDatasetDescription.suite_id: TabularDatasetDescription,
    TabularDataQuality.suite_id: TabularDataQuality,
    TextDataQuality.suite_id: TextDataQuality,
    TimeSeriesDataQuality.suite_id: TimeSeriesDataQuality,
    TimeSeriesDataset.suite_id: TimeSeriesDataset,
    TimeSeriesModelValidation.suite_id: TimeSeriesModelValidation,
    TimeSeriesMultivariate.suite_id: TimeSeriesMultivariate,
    TimeSeriesUnivariate.suite_id: TimeSeriesUnivariate,
}

# These test suites can be added by the user
custom_test_suites = {}


def _get_all_test_suites():
    """
    Returns a dictionary of all test suites.

    Merge the core and custom test suites, with the custom suites
    taking precedence, i.e. allowing overriding of core test suites
    """
    return {**core_test_suites, **custom_test_suites}


def _get_test_suite_test_ids(test_suite_class: str):
    test_ids = []

    for item in test_suite_class.tests:
        if isinstance(item, str):
            test_ids.append(item)
        elif isinstance(item, dict):
            for test_id in item["section_tests"]:
                test_ids.append(test_id)

    return test_ids


def get_by_id(test_suite_id: str):
    """
    Returns the test suite by ID
    """
    try:
        return _get_all_test_suites()[test_suite_id]
    except KeyError:
        raise ValueError(f"Test suite with ID: '{test_suite_id}' not found")


def list_suites(pretty: bool = True):
    """
    Returns a list of all available test suites
    """

    all_test_suites = _get_all_test_suites()

    if not pretty:
        return list(all_test_suites.keys())

    table = []
    for suite_id, test_suite in all_test_suites.items():
        table.append(
            {
                "ID": suite_id,
                "Name": test_suite.__name__,
                "Description": getdoc(test_suite).strip(),
                "Tests": ", ".join(_get_test_suite_test_ids(test_suite)),
            }
        )

    return format_dataframe(pd.DataFrame(table))


def describe_suite(test_suite_id: str, verbose: bool = False) -> pd.DataFrame:
    """
    Describes a Test Suite by ID

    Args:
        test_suite_id: Test Suite ID
        verbose: If True, describe all plans and tests in the Test Suite

    Returns:
        pd.DataFrame: A formatted table with the Test Suite description
    """
    test_suite = get_by_id(test_suite_id)

    if not verbose:
        return format_dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": test_suite_id,
                        "Name": test_suite.__name__,
                        "Description": getdoc(test_suite).strip(),
                        "Tests": ", ".join(_get_test_suite_test_ids(test_suite)),
                    }
                ]
            )
        )

    tests = []

    for item in test_suite.tests:
        if isinstance(item, str):
            test = load_test(item)
            tests.append(
                {
                    "Test Suite ID": test_suite_id,
                    "Test Suite Name": test_suite.__name__,
                    "Test Suite Section": "",
                    "Test ID": item,
                    "Test Name": test.__name__,
                }
            )
        elif isinstance(item, dict):
            for test_id in item["section_tests"]:
                test = load_test(test_id)
                tests.append(
                    {
                        "Test Suite ID": test_suite_id,
                        "Test Suite Name": test_suite.__name__,
                        "Test Suite Section": item["section_id"],
                        "Test ID": test_id,
                        "Test Name": test_id_to_name(test_id),
                    }
                )
        else:
            raise ValueError(f"Invalid test suite item: {item}")

    return format_dataframe(pd.DataFrame(tests).reset_index(drop=True))


# TODO: remove this... here for backwards compatibility
describe_test_suite = describe_suite


def register_test_suite(suite_id: str, suite: TestSuite):
    """
    Registers a custom test suite
    """
    custom_test_suites[suite_id] = suite
    logger.info(f"Registered test suite: {suite_id}")
