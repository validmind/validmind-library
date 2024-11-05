# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Protocol

from validmind.logging import get_logger

from ._store import test_provider_store

logger = get_logger(__name__)


class LocalTestProviderLoadModuleError(Exception):
    """
    When the local file module can't be loaded.
    """

    pass


class LocalTestProviderLoadTestError(Exception):
    """
    When local file module was loaded but the test class can't be located.
    """

    pass


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
        self.root_folder = root_folder

    def list_tests(self):
        """List all tests in the given namespace

        Returns:
            list: A list of test IDs
        """
        test_ids = []

        directories = [p.name for p in Path(self.root_folder).iterdir() if p.is_dir()]

        for d in directories:
            for path in Path(self.root_folder).joinpath(d).glob("**/**/*.py"):
                if path.name.startswith("__") or not path.name[0].isupper():
                    continue  # skip __init__.py and other special files as well as non Test files
                test_ids.append(
                    f"{self.namespace}.{d}.{path.parent.stem}.{path.stem}"
                    if path.parent.parent.stem == d
                    else f"{self.namespace}.{d}.{path.stem}"
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
            Exception: If the test can't be imported or loaded.
        """
        test_path = f"{test_id.replace('.', '/')}.py"
        file_path = os.path.join(self.root_folder, test_path)

        logger.debug(f"Loading test {test_id} from {file_path}")

        # Check if the module uses relative imports
        with open(file_path, "r") as file:
            lines = file.readlines()

        # handle test with relative imports
        if any(line.strip().startswith("from .") for line in lines):
            logger.debug("Found relative imports, using alternative import method")

            parent_folder = os.path.dirname(file_path)
            if parent_folder not in sys.path:
                sys.path.append(os.path.dirname(parent_folder))

            try:
                module = importlib.import_module(
                    f"{os.path.basename(parent_folder)}.{test_id.split('.')[-1]}"
                )
            except Exception as e:
                # error will be handled/re-raised by `load_test` func
                raise LocalTestProviderLoadModuleError(
                    f"Failed to load the module from {file_path}. Error: {str(e)}"
                )

        else:
            try:
                spec = importlib.util.spec_from_file_location(test_id, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                # error will be handled/re-raised by `load_test` func
                raise LocalTestProviderLoadModuleError(
                    f"Failed to load the module from {file_path}. Error: {str(e)}"
                )

        try:
            # find the test class that matches the last part of the test_id
            return getattr(module, test_id.split(".")[-1])
        except AttributeError as e:
            raise LocalTestProviderLoadTestError(
                f"Failed to find the test class in the module. Error: {str(e)}"
            )


class ValidMindTestProvider:
    """Test provider for ValidMind tests"""

    def __init__(self):
        # two subproviders: unit_metrics and normal tests
        self.metrics_provider = LocalTestProvider(
            os.path.join(os.path.dirname(__file__), "..", "unit_metrics")
        )
        self.tests_provider = LocalTestProvider(os.path.dirname(__file__))

    def list_tests(self) -> List[str]:
        return self.metrics_provider.list_tests() + self.tests_provider.list_tests()

    def load_test(self, test_id: str) -> callable:
        return (
            self.metrics_provider.load_test(test_id)
            if test_id.startswith("validmind.unit_metrics")
            else self.tests_provider.load_test(test_id)
        )


def register_test_provider(namespace: str, test_provider: "TestProvider") -> None:
    """Register an external test provider

    Args:
        namespace (str): The namespace of the test provider
        test_provider (TestProvider): The test provider
    """
    test_provider_store.register_test_provider(namespace, test_provider)
