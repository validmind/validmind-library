# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import asyncio

from ...logging import get_logger
from ...utils import is_notebook, run_async, run_async_check
from ..html_progress import HTMLBox, HTMLLabel, HTMLProgressBar
from .summary import TestSuiteSummary
from .test_suite import TestSuite

logger = get_logger(__name__)


class TestSuiteRunner:
    """
    Runs a test suite.
    """

    suite: TestSuite = None
    config: dict = None

    _test_configs: dict = None

    # HTML-based progress components
    html_pbar: HTMLProgressBar = None
    html_pbar_description: HTMLLabel = None
    html_pbar_box: HTMLBox = None

    def __init__(self, suite: TestSuite, config: dict = None, inputs: dict = None):
        self.suite = suite
        self.config = config or {}

        self._load_config(inputs)

    def _load_config(self, inputs: dict = None):
        """Splits the config into a global config and test configs."""
        self._test_configs = {
            test.test_id: {"inputs": inputs or {}} for test in self.suite.get_tests()
        }

        for key, value in self.config.items():
            # If the key does not exist in the test suite, we need to
            # inform the user the config is probably wrong but we will
            # keep running all tests
            if key not in self._test_configs:
                logger.warning(
                    f"Config key '{key}' does not match a test_id in the template."
                    "\n\tEnsure you registered a content block with the correct content_id in the template"
                    "\n\tThe configuration for this test will be ignored."
                )
                continue

            # override the global config (inputs) with the test-specific config
            # TODO: better configuration would make for a better DX
            self._test_configs[key] = value

    def _start_progress_bar(self, send: bool = True):
        """
        Initializes the progress bar elements.
        """
        # TODO: make this work for when user runs only a section of the test suite
        # if we are sending then there is a task for each test and logging its result
        num_tasks = self.suite.num_tests() * 2 if send else self.suite.num_tests()

        self.html_pbar_description = HTMLLabel(value="Running test suite...")
        self.html_pbar = HTMLProgressBar(
            max_value=num_tasks, description="Running test suite..."
        )
        self.html_pbar_box = HTMLBox([self.html_pbar_description, self.html_pbar])
        self.html_pbar.display()

    def _stop_progress_bar(self):
        if self.html_pbar:
            self.html_pbar.complete()
            self.html_pbar.close()
        if self.html_pbar_description:
            self.html_pbar_description.update("Test suite complete!")

    def _update_progress_message(self, message: str):
        """Updates HTML progress bar message."""
        if self.html_pbar:
            self.html_pbar.update(self.html_pbar.value, message)
        if self.html_pbar_description:
            self.html_pbar_description.update(message)

    def _increment_progress(self):
        """Increments HTML progress bar."""
        if self.html_pbar:
            self.html_pbar.update(self.html_pbar.value + 1)

    async def _log_test_result(self, test):
        """Logs a single test result to ValidMind."""
        sending_test_message = f"Sending result to ValidMind: {test.test_id}..."
        self._update_progress_message(sending_test_message)

        try:
            await test.log_async()
        except Exception:
            failure_message = "Failed to send result to ValidMind"
            self._update_progress_message(failure_message)
            logger.error(f"Failed to log result: {test.result}")
            raise

        self._increment_progress()

    async def log_results(self):
        """Logs the results of the test suite to ValidMind.

        This method will be called after the test suite has been run and all results have been
        collected. This method will log the results to ValidMind.
        """
        sending_message = (
            f"Sending results of test suite '{self.suite.suite_id}' to ValidMind..."
        )
        self._update_progress_message(sending_message)

        tests = [test for section in self.suite.sections for test in section.tests]
        # TODO: use asyncio.gather here for better performance
        for test in tests:
            await self._log_test_result(test)

    async def _check_progress(self):
        done = False

        while not done:
            progress_complete = False
            if self.html_pbar and self.html_pbar.value >= self.html_pbar.max_value:
                progress_complete = True

            if progress_complete:
                completion_message = "Test suite complete!"

                if self.html_pbar:
                    self.html_pbar.update(self.html_pbar.max_value, completion_message)
                if self.html_pbar_description:
                    self.html_pbar_description.update(completion_message)

                done = True

            await asyncio.sleep(0.5)

    def summarize(self, show_link: bool = True):
        if not is_notebook():
            return logger.info("Test suite done...")

        collecting_message = "Collecting test results..."

        if self.html_pbar:
            self.html_pbar.update(self.html_pbar.value, collecting_message)
        if self.html_pbar_description:
            self.html_pbar_description.update(collecting_message)

        summary = TestSuiteSummary(
            title=self.suite.title,
            description=self.suite.description,
            sections=self.suite.sections,
            show_link=show_link,
        )

        from ...utils import display as vm_display

        vm_display(summary)

    def run(self, send: bool = True, fail_fast: bool = False):
        """Runs the test suite, renders the summary and sends the results to ValidMind.

        Args:
            send (bool, optional): Whether to send the results to ValidMind.
                Defaults to True.
            fail_fast (bool, optional): Whether to stop running tests after the first
                failure. Defaults to False.
        """
        self._start_progress_bar(send=send)

        for section in self.suite.sections:
            for test in section.tests:
                running_message = f"Running {test.name}"

                if self.html_pbar:
                    self.html_pbar.update(self.html_pbar.value, running_message)
                if self.html_pbar_description:
                    self.html_pbar_description.update(running_message)

                test.run(
                    fail_fast=fail_fast,
                    config=self._test_configs.get(test.test_id, {}),
                )

                if self.html_pbar:
                    self.html_pbar.update(self.html_pbar.value + 1)

        if send:
            run_async(self.log_results)
            run_async_check(self._check_progress)

        self.summarize(show_link=send)

        self._stop_progress_bar()
