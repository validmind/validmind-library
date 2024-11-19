# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import List, Protocol

from validmind.logging import get_logger

logger = get_logger(__name__)


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

    def list_tests(self):
        """List all tests in the given namespace

        Returns:
            list: A list of test IDs
        """
        test_ids = []

        directories = [p.name for p in Path(self.root_folder).iterdir() if p.is_dir()]

        for d in directories:
            for path in Path(self.root_folder).joinpath(d).glob("**/**/*.py"):
                if path.name.startswith("__"):
                    continue  # skip __init__.py and other special files

                # if the file name is capitalized or it contains a function with the
                # same name as the file, then we can assume it is a test file
                if not path.name[0].isupper():
                    with open(path, "r") as f:
                        source = f.read()
                    if not re.search(r"def\s*" + re.escape(path.stem), source):
                        continue

                test_ids.append(
                    f"{d}.{path.parent.stem}.{path.stem}"
                    if path.parent.parent.stem == d
                    else f"{d}.{path.stem}"
                )

        return test_ids

    def load_test(self, test_id: str):
        """
        Load the test identified by the given test_id.

        Args:
            test_id (str): The identifier of the test. This corresponds to the relative
            path of the python file from the root folder, with slashes replaced by dots

        Returns:
            The test class that matches the last part of the test_id.

        Raises:
            LocalTestProviderLoadModuleError: If the test module cannot be imported
            LocalTestProviderLoadTestError: If the test class cannot be found in the module
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
    """Test provider for ValidMind tests"""

    def __init__(self):
        # two subproviders: unit_metrics and normal tests
        self.metrics_provider = LocalTestProvider(
            os.path.join(os.path.dirname(__file__), "..", "unit_metrics")
        )
        self.tests_provider = LocalTestProvider(os.path.dirname(__file__))

    def list_tests(self) -> List[str]:
        """List all tests in the ValidMind test provider"""
        metric_ids = [
            f"unit_metrics.{test}" for test in self.metrics_provider.list_tests()
        ]
        test_ids = self.tests_provider.list_tests()

        return metric_ids + test_ids

    def load_test(self, test_id: str) -> callable:
        """Load a ValidMind test or unit metric"""
        return (
            self.metrics_provider.load_test(test_id.replace("unit_metrics.", ""))
            if test_id.startswith("unit_metrics.")
            else self.tests_provider.load_test(test_id)
        )
