# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Decorators for creating and registering metrics with the ValidMind Library."""

# TODO: as we move entirely to a functional approach a lot of this logic
# should be moved into the __init__ to replace the old class-based stuff

import inspect
import os
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import pandas as pd

from validmind.ai.test_descriptions import get_description_metadata
from validmind.errors import MissingRequiredTestInputError
from validmind.logging import get_logger
from validmind.vm_models import (
    Metric,
    MetricResult,
    ResultSummary,
    ResultTable,
    ResultTableMetadata,
    VMDataset,
    VMModel,
)
from validmind.vm_models.figure import (
    Figure,
    is_matplotlib_figure,
    is_plotly_figure,
    is_png_image,
)
from validmind.vm_models.test.result_wrapper import MetricResultWrapper

from ._store import test_store

logger = get_logger(__name__)


_input_type_map = {
    "dataset": VMDataset,
    "datasets": List[VMDataset],
    "model": VMModel,
    "models": List[VMModel],
}


def _inspect_signature(test_func: callable):
    inputs = {}
    params = {}

    for name, arg in inspect.signature(test_func).parameters.items():
        if name in _input_type_map:
            inputs[name] = {
                "type": _input_type_map[name],
            }
        else:
            params[name] = {
                "type": arg.annotation,
                "default": (
                    arg.default if arg.default is not inspect.Parameter.empty else None
                ),
            }

    return inputs, params


def _build_result(  # noqa: C901
    results: Union[Any, Tuple[Any, ...]],
    test_id: str,
    inputs: List[str],
    params: Dict[str, Any],
    description: str = None,
    output_template: str = None,
    generate_description: bool = True,
):
    ref_id = str(uuid4())
    figure_metadata = {
        "_type": "metric",
        "_name": test_id,
        "_ref_id": ref_id,
    }

    tables = []
    figures = []
    scalars = []

    def process_result_item(item):
        # TOOD: build out a more robust/extensible system for this
        # TODO: custom type handlers would be really cool

        # unit metrics (scalar values) - for now only one per test
        if isinstance(item, int) or isinstance(item, float):
            if scalars:
                raise ValueError("Only one unit metric may be returned per test.")
            scalars.append(item)

        # plots
        elif isinstance(item, Figure):
            figures.append(item)
        elif is_matplotlib_figure(item) or is_plotly_figure(item) or is_png_image(item):
            figures.append(
                Figure(
                    key=f"{test_id}:{len(figures) + 1}",
                    figure=item,
                    metadata=figure_metadata,
                )
            )

        # tables
        elif isinstance(item, list) or isinstance(item, pd.DataFrame):
            tables.append(ResultTable(data=item))
        elif isinstance(item, dict):
            for table_name, table in item.items():
                if not isinstance(table, list) and not isinstance(table, pd.DataFrame):
                    raise ValueError(
                        f"Invalid table format: {table_name} must be a list or DataFrame"
                    )

                tables.append(
                    ResultTable(
                        data=table,
                        metadata=ResultTableMetadata(title=table_name),
                    )
                )

        else:
            raise ValueError(f"Invalid return type: {type(item)}")

    # if the results are a tuple, process each item as a separate result
    if isinstance(results, tuple):
        for item in results:
            process_result_item(item)
    else:
        process_result_item(results)

    metric_inputs = [
        sub_i.input_id if hasattr(sub_i, "input_id") else sub_i
        for i in inputs
        for sub_i in (i if isinstance(i, list) else [i])
    ]

    return MetricResultWrapper(
        result_id=test_id,
        scalar=scalars[0] if scalars else None,
        metric=(
            MetricResult(
                key=test_id,
                ref_id=ref_id,
                value="Empty",
                summary=ResultSummary(results=tables),
            )
            if tables or figures  # if tables or figures than its a traditional metric
            else None
        ),
        figures=figures,
        result_metadata=(
            [
                get_description_metadata(
                    test_id=test_id,
                    default_description=description,
                    summary=ResultSummary(results=tables).serialize(),
                    figures=figures,
                    should_generate=generate_description,
                )
            ]
            if tables or figures
            else None
        ),
        inputs=metric_inputs,
        params=params,
        output_template=output_template,
    )


def _get_run_method(func, func_inputs, func_params):
    def run(self: Metric):
        input_kwargs = {}  # map function inputs (`dataset` etc) to actual objects
        input_ids = []  # store input_ids used so they can be logged
        for key in func_inputs.keys():
            try:
                input_kwargs[key] = getattr(self.inputs, key)
                if isinstance(input_kwargs[key], list):
                    input_ids.extend([i.input_id for i in input_kwargs[key]])
                else:
                    input_ids.append(input_kwargs[key].input_id)
            except AttributeError:
                raise MissingRequiredTestInputError(f"Missing required input: {key}.")

        param_kwargs = {
            key: self.params.get(key, func_params[key]["default"])
            for key in func_params.keys()
        }

        raw_results = func(**input_kwargs, **param_kwargs)

        self.result = _build_result(
            results=raw_results,
            test_id=self.test_id,
            description=inspect.getdoc(self),
            inputs=input_ids,
            params=param_kwargs,
            output_template=self.output_template,
            generate_description=self.generate_description,
        )

        return self.result

    return run


def _get_save_func(func, test_id):
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


def metric(func_or_id):
    """
    DEPRECATED, use @vm.test instead
    """
    # print a deprecation notice and call the test() function instead
    logger.warning(
        "The @vm.metric decorator is deprecated and will be removed in a future release. "
        "Please use @vm.test instead."
    )
    return test(func_or_id)


def test(func_or_id):
    """Decorator for creating and registering metrics with the ValidMind Library.

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

        inputs, params = _inspect_signature(func)
        description = inspect.getdoc(func)
        tasks = getattr(func, "__tasks__", [])
        tags = getattr(func, "__tags__", [])

        metric_class = type(
            func.__name__,
            (Metric,),
            {
                "run": _get_run_method(func, inputs, params),
                "required_inputs": list(inputs.keys()),
                "default_params": {k: v["default"] for k, v in params.items()},
                "__doc__": description,
                "tasks": tasks,
                "tags": tags,
            },
        )
        test_store.register_custom_test(test_id, metric_class)

        # special function to allow the function to be saved to a file
        func.save = _get_save_func(func, test_id)

        return func

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
