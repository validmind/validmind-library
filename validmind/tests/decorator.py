# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Decorators for creating and registering tests with the ValidMind Library."""

import inspect
import os
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

from validmind.logging import get_logger

from ._store import test_store
from .load import load_test

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _get_save_func(func: Callable[..., Any], test_id: str) -> Callable[..., None]:
    """Helper function to save a decorated function to a file

    Useful when a custom test function has been created inline in a notebook or
    interactive session and needs to be saved to a file so it can be added to a
    test library.
    """

    # get og source before its wrapped by the test decorator
    source = inspect.getsource(func)
    # remove decorator line
    source = source.split("\n", 1)[1]

    def save(root_folder: str = ".", imports: Optional[List[str]] = None) -> None:
        parts = test_id.split(".")

        if len(parts) > 1:
            path = os.path.join(root_folder, *parts[1:-1])
            test_name = parts[-1]
            new_test_id = f"<test_provider_namespace>.{'.'.join(parts[1:])}"
        else:
            path = root_folder
            test_name = parts[0]
            new_test_id = f"<test_provider_namespace>.{test_name}"

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, f"{test_name}.py")

        _source = source.replace(f"def {func.__name__}", f"def {test_name}")

        if imports:
            imports = "\n".join(imports)
            _source = f"{imports}\n\n\n{_source}"

        # add comment to the top of the file
        _source = f"""
# Saved from {func.__module__}.{func.__name__}
# Original Test ID: {test_id}
# New Test ID: {new_test_id}

{_source}
"""

        # use black to format the code
        try:
            import black

            _source = black.format_str(_source, mode=black.FileMode())
        except ImportError:
            # ignore if not available
            pass

        with open(full_path, "w") as file:
            file.writelines(_source)

        logger.info(
            f"Saved to {os.path.abspath(full_path)}!"
            "Be sure to add any necessary imports to the top of the file."
        )
        logger.info(
            f"This metric can be run with the ID: {new_test_id}",
        )

    return save


def test(func_or_id: Union[Callable[..., Any], str, None]) -> Callable[[F], F]:
    """Decorator for creating and registering custom tests

    This decorator registers the function it wraps as a test function within ValidMind
    under the provided ID. Once decorated, the function can be run using the
    `validmind.tests.run_test` function.

    The function can take two different types of arguments:

    - Inputs: ValidMind model or dataset (or list of models/datasets). These arguments
      must use the following names: `model`, `models`, `dataset`, `datasets`.
    - Parameters: Any additional keyword arguments of any type (must have a default
      value) that can have any name.

    The function should return one of the following types:

    - Table: Either a list of dictionaries or a pandas DataFrame
    - Plot: Either a matplotlib figure or a plotly figure
    - Scalar: A single number (int or float)
    - Boolean: A single boolean value indicating whether the test passed or failed

    The function may also include a docstring. This docstring will be used and logged
    as the metric's description.

    Args:
        func_or_id (Union[Callable[..., Any], str, None]): Either the function to decorate
            or the test ID. If None, the function name is used.

    Returns:
        Callable[[F], F]: The decorated function.
    """

    def decorator(func: F) -> F:
        test_id = func_or_id or f"validmind.custom_metrics.{func.__name__}"
        test_func = load_test(test_id, func, reload=True)
        test_store.register_test(test_id, test_func)

        # special function to allow the function to be saved to a file
        save_func = _get_save_func(func, test_id)

        wrapper = wraps(func)(test_func)
        wrapper.test_id = test_id
        wrapper.save = save_func

        return wrapper

    if callable(func_or_id):
        return decorator(func_or_id)

    return decorator


def tasks(*tasks: str) -> Callable[[F], F]:
    """Decorator for specifying the task types that a test is designed for.

    Args:
        *tasks: The task types that the test is designed for.
    """

    def decorator(func: F) -> F:
        func.__tasks__ = list(tasks)
        return func

    return decorator


def tags(*tags: str) -> Callable[[F], F]:
    """Decorator for specifying tags for a test.

    Args:
        *tags: The tags to apply to the test.
    """

    def decorator(func: F) -> F:
        func.__tags__ = list(tags)
        return func

    return decorator
