# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Decorators for creating and registering tests with the ValidMind Library."""

import inspect
import os
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

from validmind.logging import get_logger

from ._store import scorer_store, test_store
from .load import _inspect_signature, load_test

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


def scorer(func_or_id: Union[Callable[..., Any], str, None] = None) -> Callable[[F], F]:
    """Decorator for creating and registering custom scorers

    This decorator registers the function it wraps as a scorer function within ValidMind
    under the provided ID. Once decorated, the function can be run using the
    `validmind.scorer.run_scorer` function.

    The scorer ID can be provided in three ways:
    1. Explicit ID: `@scorer("validmind.scorer.classification.BrierScore")`
    2. Auto-generated from path: `@scorer()` - automatically generates ID from file path
    3. Function name only: `@scorer` - uses function name with validmind.scorer prefix

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
    - List: A list of values (for row-level metrics) or a list of dictionaries with consistent keys
    - Any other type: The output will be stored as raw data for use by calling code

    When returning a list of dictionaries:
    - All dictionaries must have the same keys
    - The list length must match the number of rows in the dataset
    - Each dictionary key will become a separate column when using assign_scores
    - Column naming follows the pattern: {model_id}_{metric_name}_{dict_key}

    Note: Scorer outputs are not logged to the backend and are intended for use
    by other parts of the system (e.g., assign_scores method).

    The function may also include a docstring. This docstring will be used and logged
    as the scorer's description.

    Args:
        func_or_id (Union[Callable[..., Any], str, None], optional): Either the function to decorate
            or the scorer ID. If None or empty string, the ID is auto-generated from the file path.
            Defaults to None.

    Returns:
        Callable[[F], F]: The decorated function.
    """

    def decorator(func: F) -> F:
        # Determine the scorer ID
        if func_or_id is None or func_or_id == "":
            # Auto-generate ID from file path
            scorer_id = _generate_scorer_id_from_path(func)
        elif isinstance(func_or_id, str):
            scorer_id = func_or_id
        else:
            # func_or_id is the function itself, auto-generate ID
            scorer_id = _generate_scorer_id_from_path(func)

        # Don't call load_test during registration to avoid circular imports
        # Just register the function directly in the scorer store
        # Scorers should only be stored in the scorer store, not the test store
        scorer_store.register_scorer(scorer_id, func)

        # special function to allow the function to be saved to a file
        save_func = _get_save_func(func, scorer_id)

        # Add attributes to the function
        func.scorer_id = scorer_id
        func.save = save_func
        func._is_scorer = True  # Mark this function as a scorer

        func.inputs, func.params = _inspect_signature(func)

        return func

    if callable(func_or_id):
        return decorator(func_or_id)
    elif func_or_id is None:
        # Handle @scorer() case - return decorator that will auto-generate ID
        return decorator

    return decorator


def _generate_scorer_id_from_path(func: Callable[..., Any]) -> str:
    """Generate a scorer ID from the function's file path.

    This function automatically generates a scorer ID based on the file path
    where the function is defined, following the same pattern as the test system.

    Args:
        func: The function to generate an ID for

    Returns:
        str: The generated scorer ID in the format validmind.scorer.path.to.function
    """
    import inspect

    try:
        # Get the file path of the function
        file_path = inspect.getfile(func)

        # Find the scorer directory in the path
        scorer_dir = os.path.join(os.path.dirname(__file__), "..", "scorer")
        scorer_dir = os.path.abspath(scorer_dir)

        # Get relative path from scorer directory
        try:
            rel_path = os.path.relpath(file_path, scorer_dir)
        except ValueError:
            # If file is not under scorer directory, fall back to function name
            return f"validmind.scorer.{func.__name__}"

        # Convert path to scorer ID
        # Remove .py extension and replace path separators with dots
        scorer_path = os.path.splitext(rel_path)[0].replace(os.sep, ".")

        # If the path is just the filename (no subdirectories), use it as is
        if scorer_path == func.__name__:
            return f"validmind.scorer.{func.__name__}"

        # Otherwise, use the full path
        return f"validmind.scorer.{scorer_path}"

    except (OSError, TypeError):
        # Fallback to function name if we can't determine the path
        return f"validmind.scorer.{func.__name__}"
