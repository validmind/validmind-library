# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Protocol

from validmind.logging import get_logger

logger = get_logger(__name__)

# list all files in directory of this file
__private_files = [f.name for f in Path(__file__).parent.glob("*.py")]


def _is_test_file(path: Path) -> bool:
    return (
        path.name[0].isupper()
        or re.search(r"def\s*" + re.escape(path.stem), path.read_text())
    ) and path.name not in __private_files


class TestProvider(Protocol):
    """Protocol for user-defined test providers"""

    def list_tests(self) -> List[str]:
        """List all tests in the given namespace

        Returns:
            list: A list of test IDs
        """
        ...

    def load_test(self, test_id: str) -> callable:
        """Load the test function identified by the given test_id

        Args:
            test_id (str): The test ID (does not contain the namespace under which
                the test is registered)

        Returns:
            callable: The test function

        Raises:
            FileNotFoundError: If the test is not found
        """
        ...


class LocalTestProvider:
    """
    Test providers in ValidMind are responsible for loading tests from different sources,
    such as local files, databases, or remote services. The LocalTestProvider specifically
    loads tests from the local file system.

    To use the LocalTestProvider, you need to provide the root_folder, which is the
    root directory for local tests. The test_id is a combination of the namespace (set
    when registering the test provider) and the path to the test class module, where
    slashes are replaced by dots and the .py extension is left out.

    Example usage:

    ```
    # Create an instance of LocalTestProvider with the root folder
    test_provider = LocalTestProvider("/path/to/tests/folder")

    # Register the test provider with a namespace
    register_test_provider("my_namespace", test_provider)

    # List all tests in the namespace (returns a list of test IDs)
    test_provider.list_tests()
    # this is used by the validmind.tests.list_tests() function to aggregate all tests
    # from all test providers

    # Load a test using the test_id (namespace + path to test class module)
    test = test_provider.load_test("my_namespace.my_test_class")
    # full path to the test class module is /path/to/tests/folder/my_test_class.py
    ```

    Attributes:
        root_folder (str): The root directory for local tests.
    """

    def __init__(self, root_folder: str):
        """
        Initialize the LocalTestProvider with the given root_folder
        (see class docstring for details)

        Args:
            root_folder (str): The root directory for local tests.
        """
        self.root_folder = os.path.abspath(root_folder)

    def list_tests(self) -> List[str]:
        """List all tests in the given namespace

        Returns:
            list: A list of test IDs
        """
        test_files = []
        for root, _, files in os.walk(self.root_folder):
            for file in files:
                if not file.endswith(".py"):
                    continue

                path = Path(os.path.join(root, file))
                if _is_test_file(path):
                    rel_path = os.path.relpath(path, self.root_folder)
                    test_id = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                    test_files.append(test_id)

        return test_files

    def load_test(self, test_id: str) -> Callable[..., Any]:
        """Load the test function identified by the given test_id

        Args:
            test_id (str): The test ID (does not contain the namespace under which
                the test is registered)

        Returns:
            callable: The test function

        Raises:
            FileNotFoundError: If the test is not found
        """
        # Convert test_id to file path
        file_path = os.path.join(self.root_folder, f"{test_id.replace('.', '/')}.py")
        file_path = os.path.abspath(file_path)

        module_dir = os.path.dirname(file_path)
        module_name = test_id.split(".")[-1]

        # module specification
        spec = importlib.util.spec_from_file_location(
            name=module_name,
            location=file_path,
            submodule_search_locations=[module_dir],
        )

        # module instance from specification
        module = importlib.util.module_from_spec(spec)

        # add module to sys.modules
        sys.modules[module_name] = module
        # execute the module
        spec.loader.exec_module(module)

        # test function should match the module (file) name exactly
        return getattr(module, module_name)


class ValidMindTestProvider:
    """Provider for built-in ValidMind tests"""

    def __init__(self) -> None:
        # three subproviders: unit_metrics, scorers, and normal tests
        self.unit_metrics_provider = LocalTestProvider(
            os.path.join(os.path.dirname(__file__), "..", "unit_metrics")
        )
        self.scorers_provider = LocalTestProvider(
            os.path.join(os.path.dirname(__file__), "..", "scorer")
        )
        self.test_provider = LocalTestProvider(os.path.dirname(__file__))

    def list_tests(self) -> List[str]:
        """List all tests in the given namespace (excludes scorers)"""
        unit_metric_ids = [
            f"unit_metrics.{test}" for test in self.unit_metrics_provider.list_tests()
        ]
        # Exclude scorers from general test list - they have their own list_scorers() function
        test_ids = self.test_provider.list_tests()

        return unit_metric_ids + test_ids

    def load_test(self, test_id: str) -> Callable[..., Any]:
        """Load the test function identified by the given test_id"""
        if test_id.startswith("unit_metrics."):
            return self.unit_metrics_provider.load_test(
                test_id.replace("unit_metrics.", "")
            )
        elif test_id.startswith("scorer."):
            return self.scorers_provider.load_test(test_id.replace("scorer.", ""))
        else:
            return self.test_provider.load_test(test_id)
