"""__TEST_NAME__ Threshold Test"""

from dataclasses import dataclass
from typing import List

from validmind.logging import get_logger
from validmind.vm_models import (
    ResultSummary,
    ResultTable,
    ResultTableMetadata,
    TestPlanTestResult,
    TestResult,
    ThresholdTest,
)

logger = get_logger(__name__)


@dataclass
class __TEST_NAME__(ThresholdTest):
    """
    Test that the model's prediction accuracy on a dataset meets or
    exceeds a predefined threshold.
    """

    category = "__TEST_CATEGORY__"  # model_performance, data_quality, etc.
    name = "__TEST_ID__"
    required_context = []
    default_params = {}

    def run(self) -> TestPlanTestResult:
        """Run the test and cache the results

        Returns:
            TestPlanTestResult: The results of the test.
        """
        return self.cache_results(
            test_results_list=[
                TestResult(
                    passed=True,  # whether this test passed
                    values={
                        "hello": "world",
                    },
                )
            ],
            passed=True,  # whether all tests passed
            figures=None,  # return a figure by importing from validmind.vm_models
        )

    def summary(self, results: List[TestResult], all_passed: bool) -> ResultSummary:
        """Summarize the results of the test.

        Args:
            results (List[TestResult]): The results of the test.
            all_passed (bool): Whether all tests passed.

        Returns:
            ResultSummary: A summary of the test results.
        """
        return ResultSummary(
            results=[
                ResultTable(
                    data=results[0].values,
                    metadata=ResultTableMetadata(
                        title="__TEST_NAME__ Test Results",
                    ),
                )
            ]
        )