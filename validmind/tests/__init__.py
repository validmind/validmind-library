# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""ValidMind Tests Module"""

from ..errors import LoadTestError
from ..logging import get_logger
from .decorator import metric, tags, tasks, test
from .load import describe_test, list_tests, load_test
from .metadata import list_tags, list_tasks, list_tasks_and_tags
from .run import run_test
from .test_providers import TestProvider, register_test_provider

logger = get_logger(__name__)

__all__ = [
    "data_validation",
    "model_validation",
    "prompt_validation",
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
    "metric",  # DEPRECATED
    "tags",
    "tasks",
]
