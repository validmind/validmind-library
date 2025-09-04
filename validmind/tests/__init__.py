# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""ValidMind Tests Module"""

from ..errors import LoadTestError
from ..logging import get_logger
from ._store import test_provider_store
from .decorator import scorer, tags, tasks, test
from .load import (
    describe_test,
    list_tags,
    list_tasks,
    list_tasks_and_tags,
    list_tests,
    load_test,
)
from .run import run_test
from .test_providers import LocalTestProvider, TestProvider

logger = get_logger(__name__)


# public-facing function for registering test providers
def register_test_provider(namespace: str, test_provider: TestProvider) -> None:
    """Register an external test provider

    Args:
        namespace (str): The namespace of the test provider
        test_provider (TestProvider): The test provider
    """
    if not hasattr(test_provider, "list_tests"):
        raise ValueError("test_provider must implement `list_tests` method")

    if not hasattr(test_provider, "load_test"):
        raise ValueError("test_provider must implement `load_test` method")

    test_provider_store.register_test_provider(namespace, test_provider)


__all__ = [
    "data_validation",
    "model_validation",
    "prompt_validation",
    "plots",
    "stats",
    "list_tests",
    "load_test",
    "describe_test",
    "run_test",
    "register_test_provider",
    "LoadTestError",
    "LocalTestProvider",
    "TestProvider",
    # Metadata
    "list_tags",
    "list_tasks",
    "list_tasks_and_tags",
    # Decorators for functional metrics
    "test",
    "scorer",
    "tags",
    "tasks",
]
