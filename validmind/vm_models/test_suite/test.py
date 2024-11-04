# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from ...errors import should_raise_on_fail_fast
from ...logging import get_logger, log_performance
from ...tests import LoadTestError
from ...tests import load_test as load_test_class
from ...utils import test_id_to_name
from ..result import FailedResultWrapper, ResultWrapper
from ..test.test import Test
from ..test_context import TestContext, TestInput

logger = get_logger(__name__)


class TestSuiteTest:
    """
    Wraps a 'Test' in a Test Suite and handles logic and state for that test
    """

    test_id: str
    output_template: str = None
    name: str = None

    _test_class: Test = None
    _test_instance: Test = None

    result: object = None

    def __init__(self, test_id_or_obj):
        """Load the test class from the test id

        Args:
            test_id_or_obj (str): The test id or a dict with test id and other options
        """
        if isinstance(test_id_or_obj, str):
            self.test_id = test_id_or_obj
        else:
            self.test_id = test_id_or_obj["id"]
            self.output_template = test_id_or_obj.get("output_template")

        self.name = test_id_to_name(self.test_id)

        try:
            self._test_class = load_test_class(self.test_id)
        except LoadTestError as e:
            self.result = FailedResultWrapper(
                error=e,
                message=f"Failed to load test '{self.test_id}'",
                result_id=self.test_id,
            )
        except Exception as e:
            # The test suite runner will appropriately ignore this error
            # since _test_class is None
            logger.error(f"Failed to load test '{self.test_id}': {e}")

    @property
    def test_type(self):
        return self._test_class.test_type

    def get_default_params(self):
        """Returns the default params for the test"""
        if not self._test_class:
            return {}

        return self._test_class.default_params

    def load(self, inputs: TestInput, context: TestContext, config: dict = None):
        """Load an instance of the test class"""
        if not self._test_class:
            return

        try:
            self._test_instance = self._test_class(
                test_id=self.test_id,
                context=context,
                inputs=inputs,
                params=config,
                output_template=self.output_template,
            )
        except Exception as e:
            logger.error(
                f"Failed to load test '{self.test_id}': "
                f"({e.__class__.__name__}) {e}"
            )
            self.result = FailedResultWrapper(
                error=e,
                message=f"Failed to load test '{self.name}'",
                result_id=self.test_id,
            )

    def run(self, fail_fast: bool = False):
        """Run the test"""
        if not self._test_instance:
            # test failed to load and we have already logged the error
            return

        try:
            self._test_instance.validate_inputs()

            # run the test and log the performance if LOG_LEVEL is set to DEBUG
            log_performance(
                func=self._test_instance.run,
                name=self.test_id,
                logger=logger,
            )()  # this is a decorator so we need to call it

        except Exception as e:
            if fail_fast and should_raise_on_fail_fast(e):
                raise e  # Re-raise the exception if we are in fail fast mode

            logger.error(
                f"Failed to run test '{self.test_id}': " f"({e.__class__.__name__}) {e}"
            )
            self.result = FailedResultWrapper(
                name=f"Failed {self._test_instance.test_type}",
                error=e,
                message=f"Failed to run '{self.name}'",
                result_id=self.test_id,
            )

            return

        if self._test_instance.result is None:
            self.result = FailedResultWrapper(
                name=f"Failed {self._test_instance.test_type}",
                error=None,
                message=f"'{self.name}' did not return a result",
                result_id=self.test_id,
            )

            return

        if not isinstance(self._test_instance.result, ResultWrapper):
            self.result = FailedResultWrapper(
                name=f"Failed {self._test_instance.test_type}",
                error=None,
                message=f"{self.name} returned an invalid result: {self._test_instance.result}",
                result_id=self.test_id,
            )

            return

        self.result = self._test_instance.result

    async def log_async(self):
        """Log the result for this test to ValidMind"""
        if not self.result:
            raise ValueError("Cannot log test result before running the test")

        await self.result.log_async()
