# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Decorators for creating and registering metrics with the ValidMind framework."""

import inspect
import os
from functools import wraps

from validmind.logging import get_logger

from ._store import test_store

logger = get_logger(__name__)


def _get_save_func(func, test_id):
    """Helper function to save a decorated function to a file

    Useful when a custom test function has been created inline in a notebook or
    interactive session and needs to be saved to a file so it can be added to a
    test library.
    """

    def save(root_folder=".", imports=None):
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

        source = inspect.getsource(func)
        # remove decorator line
        source = source.split("\n", 1)[1]
        if imports:
            imports = "\n".join(imports)
            source = f"{imports}\n\n\n{source}"
        # add comment to the top of the file
        source = f"""
# Saved from {func.__module__}.{func.__name__}
# Original Test ID: {test_id}
# New Test ID: {new_test_id}

{source}
"""

        # ensure that the function name matches the test name
        source = source.replace(f"def {func.__name__}", f"def {test_name}")

        # use black to format the code
        try:
            import black

            source = black.format_str(source, mode=black.FileMode())
        except ImportError:
            # ignore if not available
            pass

        with open(full_path, "w") as file:
            file.writelines(source)

        logger.info(
            f"Saved to {os.path.abspath(full_path)}!"
            "Be sure to add any necessary imports to the top of the file."
        )
        logger.info(
            f"This metric can be run with the ID: {new_test_id}",
        )

    return save


def test(func_or_id):
    """Decorator for creating and registering metrics with the ValidMind framework.

    Creates a metric object and registers it with ValidMind under the provided ID. If
    no ID is provided, the function name will be used as to build one. So if the
    function name is `my_metric`, the metric will be registered under the ID
    `validmind.custom_metrics.my_metric`.

    This decorator works by creating a new `Metric` class will be created whose `run`
    method calls the decorated function. This function should take as arguments the
    inputs it requires (`dataset`, `datasets`, `model`, `models`) followed by any
    parameters. It can return any number of the following types:

    - Table: Either a list of dictionaries or a pandas DataFrame
    - Plot: Either a matplotlib figure or a plotly figure
    - Scalar: A single number or string

    The function may also include a docstring. This docstring will be used and logged
    as the metric's description.

    Args:
        func: The function to decorate
        test_id: The identifier for the metric. If not provided, the function name is used.

    Returns:
        The decorated function.
    """

    def decorator(func):
        test_id = func_or_id or f"validmind.custom_metrics.{func.__name__}"
        test_store.register_test(test_id, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # special function to allow the function to be saved to a file
        wrapper.save = _get_save_func(func, test_id)

        return wrapper

    if callable(func_or_id):
        return decorator(func_or_id)

    return decorator


def tasks(*tasks):
    """Decorator for specifying the task types that a metric is designed for.

    Args:
        *tasks: The task types that the metric is designed for.
    """

    def decorator(func):
        func.__tasks__ = list(tasks)
        return func

    return decorator


def tags(*tags):
    """Decorator for specifying tags for a metric.

    Args:
        *tags: The tags to apply to the metric.
    """

    def decorator(func):
        func.__tags__ = list(tags)
        return func

    return decorator
