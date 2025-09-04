# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Module for storing loaded tests and test providers"""


from typing import Any, Callable, Optional

from .test_providers import TestProvider, ValidMindTestProvider


def singleton(cls):
    """Decorator to make a class a singleton"""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class TestProviderStore:
    """Singleton class for storing test providers"""

    def __init__(self):
        self.test_providers = {}

    def has_test_provider(self, namespace: str) -> bool:
        """Check if a test provider exists by namespace

        Args:
            namespace (str): The namespace of the test provider

        Returns:
            bool: True if the test provider exists
        """
        return namespace in self.test_providers

    def get_test_provider(self, namespace: str) -> TestProvider:
        """Get a test provider by namespace

        Args:
            namespace (str): The namespace of the test provider

        Returns:
            TestProvider: The test provider
        """
        return self.test_providers.get(namespace)

    def register_test_provider(self, namespace: str, test_provider) -> None:
        """Register an external test provider

        Args:
            namespace (str): The namespace of the test provider
            test_provider (TestProvider): The test provider
        """
        self.test_providers[namespace] = test_provider


class TestStore:
    """Singleton class for storing loaded tests"""

    def __init__(self):
        self.tests = {}

    def get_test(self, test_id: str) -> Optional[Callable[..., Any]]:
        """Get a test by test ID

        Args:
            test_id (str): The test ID

        Returns:
            Optional[Callable[..., Any]]: The test function if found, None otherwise
        """
        return self.tests.get(test_id)

    def register_test(
        self, test_id: str, test: Optional[Callable[..., Any]] = None
    ) -> None:
        """Register a test

        Args:
            test_id (str): The test ID
            test (Optional[Callable[..., Any]], optional): The test function. Defaults to None.
        """
        self.tests[test_id] = test


@singleton
class ScorerStore:
    """Singleton class for storing loaded scorers"""

    def __init__(self):
        self.scorers = {}

    def get_scorer(self, scorer_id: str) -> Optional[Callable[..., Any]]:
        """Get a scorer by scorer ID

        Args:
            scorer_id (str): The scorer ID

        Returns:
            Optional[Callable[..., Any]]: The scorer function if found, None otherwise
        """
        return self.scorers.get(scorer_id)

    def register_scorer(
        self, scorer_id: str, scorer: Optional[Callable[..., Any]] = None
    ) -> None:
        """Register a scorer

        Args:
            scorer_id (str): The scorer ID
            scorer (Optional[Callable[..., Any]], optional): The scorer function. Defaults to None.
        """
        self.scorers[scorer_id] = scorer


test_store = TestStore()
scorer_store = ScorerStore()
test_provider_store = TestProviderStore()

# setup built-in test providers
test_provider_store.register_test_provider("validmind", ValidMindTestProvider())
