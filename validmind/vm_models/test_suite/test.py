# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, Union

from ...errors import LoadTestError, should_raise_on_fail_fast
from ...logging import get_logger, log_performance
from ...tests.run import run_test
from ...utils import test_id_to_name
from ..result import ErrorResult, Result, TestResult

logger = get_logger(__name__)


class TestSuiteTest:
    """
    Wraps a 'Test' in a Test Suite and handles logic and state for that test.
    """

    test_id: str
    output_template: str = None
    name: Union[str, None] = None
    description: Union[Dict[str, Any], None] = None
    result: Union[Result, None] = None

    _load_failed: bool = False

    def __init__(self, test_id_or_obj):
        """Load the test class from the test ID.

        Args:
            test_id_or_obj (str): The test ID or a dict with test ID and other options.
        """
        if isinstance(test_id_or_obj, str):
            self.test_id = test_id_or_obj
        else:
            self.test_id = test_id_or_obj["id"]
            self.output_template = test_id_or_obj.get("output_template")

        self.name = test_id_to_name(self.test_id)

    def get_default_config(self):
        """Returns the default configuration for the test."""
        from ...tests.load import load_test

        try:
            test_func = load_test(self.test_id)
        except LoadTestError as e:
            logger.error(f"Failed to load test '{self.test_id}': {e}")

            self._load_failed = True
            self.result = ErrorResult(
                error=e,
                message=f"Failed to load test '{self.name}'",
                result_id=self.test_id,
            )

            return None

        config = {
            # we use the input name ('dataset', 'model') as the key and the value
            "inputs": {k: k for k in test_func.inputs},
            "params": {k: v.get("default") for k, v in test_func.params.items()},
        }

        return config

    def run(self, fail_fast: bool = False, config: dict = None):
        """Run the test."""
        if self._load_failed:
            return

        try:
            # run the test and log the performance if LOG_LEVEL is set to DEBUG
            @log_performance(name=self.test_id, logger=logger)
            def run_test_with_logging():
                return run_test(
                    self.test_id,
                    **(config or {}),
                    show=False,
                )

            self.result = run_test_with_logging()

        except Exception as e:
            if fail_fast and should_raise_on_fail_fast(e):
                raise e  # Re-raise the exception if we are in fail fast mode

            logger.error(
                f"Failed to run test '{self.test_id}': " f"({e.__class__.__name__}) {e}"
            )
            self.result = ErrorResult(
                error=e,
                message=f"Failed to run '{self.name}'",
                result_id=self.test_id,
            )

        if self.result is None:
            self.result = ErrorResult(
                error=None,
                message=f"'{self.name}' did not return a result",
                result_id=self.test_id,
            )

        if not isinstance(self.result, Result):
            self.result = ErrorResult(
                error=None,
                message=f"{self.name} returned an invalid result: {self._test_instance.result}",
                result_id=self.test_id,
            )

    async def log_async(self):
        """Log the result for this test to ValidMind."""
        if not self.result:
            raise ValueError("Cannot log test result before running the test")

        if isinstance(self.result, TestResult):
            return await self.result.log_async()
