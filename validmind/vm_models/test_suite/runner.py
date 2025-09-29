# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import ipywidgets as widgets
from IPython.display import display

from ...logging import get_logger
from ...utils import is_notebook
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

    pbar: widgets.IntProgress = None
    pbar_description: widgets.Label = None
    pbar_box: widgets.HBox = None

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

        self.pbar_description = widgets.Label(value="Running test suite...")
        self.pbar = widgets.IntProgress(max=num_tasks, orientation="horizontal")
        self.pbar_box = widgets.HBox([self.pbar_description, self.pbar])

        display(self.pbar_box)

    def _stop_progress_bar(self):
        self.pbar_description.value = "Test suite complete!"
        self.pbar.close()

    def log_results(self):
        """Logs the results of the test suite to ValidMind.

        This method will be called after the test suite has been run and all results have been
        collected. This method will log the results to ValidMind.
        """
        self.pbar_description.value = (
            f"Sending results of test suite '{self.suite.suite_id}' to ValidMind..."
        )

        tests = [test for section in self.suite.sections for test in section.tests]
        for test in tests:
            self.pbar_description.value = (
                f"Sending result to ValidMind: {test.test_id}..."
            )

            try:
                test.log_sync()
            except Exception as e:
                self.pbar_description.value = "Failed to send result to ValidMind"
                logger.error(f"Failed to log result: {test.result}")

                raise e

            self.pbar.value += 1

    def _check_progress(self):
        if self.pbar.value == self.pbar.max:
            self.pbar_description.value = "Test suite complete!"

    def summarize(self, show_link: bool = True):
        if not is_notebook():
            return logger.info("Test suite done...")

        self.pbar_description.value = "Collecting test results..."

        summary = TestSuiteSummary(
            title=self.suite.title,
            description=self.suite.description,
            sections=self.suite.sections,
            show_link=show_link,
        )
        summary.display()

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
                self.pbar_description.value = f"Running {test.name}"
                test.run(
                    fail_fast=fail_fast,
                    config=self._test_configs.get(test.test_id, {}),
                )
                self.pbar.value += 1

        if send:
            self.log_results()
            self._check_progress()

        self.summarize(show_link=send)

        self._stop_progress_bar()
