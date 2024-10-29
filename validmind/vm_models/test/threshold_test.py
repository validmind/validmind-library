# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
(Threshold)Test class wrapper. Our API exposes the concept of of a
Test (as test_results) but we'll refer to it as a ThresholdTest to
avoid confusion with the "tests" in the general data science/modeling sense.
"""

from dataclasses import dataclass
from typing import ClassVar, List, Optional

from ...ai.test_descriptions import get_description_metadata
from ..figure import Figure
from .result_summary import ResultSummary, ResultTable
from .result_wrapper import ThresholdTestResultWrapper
from .test import Test
from .threshold_test_result import ThresholdTestResult, ThresholdTestResults


@dataclass
class ThresholdTest(Test):
    """
    A threshold test is a combination of a metric/plot we track and a
    corresponding set of parameters and thresholds values that allow
    us to determine whether the metric/plot passes or fails.
    """

    # Class Variables
    test_type: ClassVar[str] = "ThresholdTest"
    category: ClassVar[str]  # should be overridden by test classes

    # Instance Variables
    result: ThresholdTestResults = None  # populated by cache_results() method

    def summary(self, results: Optional[List[ThresholdTestResult]], all_passed: bool):
        """
        Return the threshold test summary. May be overridden by subclasses. Defaults to showing
        a table with test_name(optional), column and passed.

        The test summary allows renderers (e.g. Word and ValidMind UI) to display a
        short summary of the test results.
        """
        if results is None:
            return None

        results_table = []
        for test_result in results:
            result_object = {
                "passed": test_result.passed,
            }

            if test_result.test_name is not None:
                result_object["test_name"] = test_result.test_name
            if test_result.column is not None:
                result_object["column"] = test_result.column

            results_table.append(result_object)

        return ResultSummary(results=[ResultTable(data=results_table)])

    def cache_results(
        self,
        test_results_list: List[ThresholdTestResult],
        passed: bool,
        figures: Optional[List[Figure]] = None,
    ):
        """
        Cache the individual results of the threshold test as a list of ThresholdTestResult objects

        Args:
            result (List[ThresholdTestResult]): The results of the threshold test
            passed (bool): Whether the threshold test passed or failed

        Returns:
            TestSuiteResult: The test suite result object
        """
        result_summary = self.summary(test_results_list, passed)

        self.result = ThresholdTestResultWrapper(
            result_id=self.test_id,
            result_description=self.description(),
            result_metadata=[
                get_description_metadata(
                    test_id=self.test_id,
                    default_description=self.description(),
                    summary=result_summary.serialize(),
                    figures=figures,
                    prefix="test_description",
                    should_generate=self.generate_description,
                )
            ],
            inputs=self.get_accessed_inputs(),
            test_results=ThresholdTestResults(
                test_name=self.test_id,
                ref_id=self._ref_id,
                params=self.params,
                passed=passed,
                results=test_results_list,
                summary=result_summary,
            ),
            figures=figures,
        )

        return self.result
