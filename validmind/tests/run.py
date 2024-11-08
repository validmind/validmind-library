# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from inspect import getdoc
from itertools import product
from typing import Any, Dict, List, Union

from validmind.errors import MissingRequiredTestInputError
from validmind.input_registry import input_registry
from validmind.logging import get_logger
from validmind.vm_models.input import VMInput
from validmind.vm_models.result import TestResult, build_test_result

from .__types__ import TestID
from .load import load_test

logger = get_logger(__name__)


def _cartesian_product(grid: Dict[str, List[Any]]):
    """Get all possible combinations for a grid of inputs or params

    Example:
        _cartesian_product({"a": [1, 2], "b": [3, 4]})
        >>> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def _get_test_kwargs(test_func, inputs, params):
    input_kwargs = {}  # map function inputs (`dataset` etc) to actual objects

    for key in test_func.inputs.keys():
        try:
            _input = inputs[key]
        except KeyError:
            raise MissingRequiredTestInputError(f"Missing required input: {key}.")

        # 1) retrieve input object from input registry if an input_id string is provided
        # 2) check the input_id type if a list of inputs (mix of strings and objects) is provided
        # 3) if its a dict, it should contain the `input_id` key as well as other options
        if isinstance(_input, str):
            _input = input_registry.get(key=_input)
        elif isinstance(_input, list) or isinstance(_input, tuple):
            _input = [
                input_registry.get(key=v) if isinstance(v, str) else v for v in _input
            ]
        elif isinstance(_input, dict):
            assert "input_id" in _input, (
                "Input dictionary must contain an 'input_id' key "
                "to retrieve the input object from the input registry."
            )
            _input = input_registry.get(key=_input["input_id"]).with_options(
                **{k: v for k, v in _input.items() if k != "input_id"}
            )

        input_kwargs[key] = _input

    param_kwargs = {
        key: params.get(key, test_func.params[key]["default"])
        for key in test_func.params.keys()
    }

    return input_kwargs, param_kwargs


def _combine_tables(results: List[TestResult]):
    tables = []

    for result in results:
        tables.extend(result.tables)

    return tables


def _combine_figures(results: List[TestResult]):
    figures = []

    for result in results:
        figures.extend(result.figures)

    return figures


def _run_composite_test(
    test_id: TestID,
    metric_ids: List[TestID],
    inputs: Dict[str, Union[VMInput, List[VMInput]]],
    params: Dict[str, Any],
    show: bool = True,
    generate_description: bool = True,
):
    results = [
        run_test(
            test_id=metric_id,
            inputs=inputs,
            params=params,
            show=show,
            generate_description=generate_description,
        )
        for metric_id in metric_ids
    ]

    result = build_test_result(
        outputs=[result.metric for result in results],
        test_id=test_id,
        inputs=inputs,
        params=params,
        description="\n---\n".join([result.description for result in results]),
        generate_description=generate_description,
    )

    if show:
        result.show()

    return result


def _run_comparison_test(
    test_id: TestID,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
    inputs: Dict[str, Union[VMInput, List[VMInput]]],
    params: Dict[str, Any],
    show: bool = True,
    generate_description: bool = True,
):
    if inputs:
        input_grid = _cartesian_product(inputs)

    if params:
        param_grid = _cartesian_product(params)

    full_grid = _cartesian_product(input_grid, param_grid)

    print(full_grid)

    results = [
        run_test(
            test_id=test_id,
            inputs=group,
            params=params,
            show=show,
            generate_description=generate_description,
        )
        for group in full_grid
    ]

    combined_tables = _combine_tables(results)
    combined_figures = _combine_figures(results)

    combined_outputs = tuple(*combined_tables, *combined_figures)

    result = build_test_result(
        outputs=combined_outputs,
        test_id=test_id,
        inputs=inputs,
        params=params,
        description=results[0].description,
        generate_description=generate_description,
    )

    if show:
        result.show()

    return result


def run_test(
    test_id: Union[TestID, None] = None,
    params: Union[Dict[str, Any], None] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    inputs: Union[Dict[str, Any], None] = None,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    name: Union[str, None] = None,
    unit_metrics: Union[List[TestID], None] = None,
    show: bool = True,
    generate_description: bool = True,
    **kwargs,
) -> TestResult:
    """Run a ValidMind or custom test

    This function is the main entry point for running tests. It can run simple unit metrics,
    ValidMind and custom tests, composite tests made up of multiple unit metrics and comparison
    tests made up of multiple tests.

    Args:
        test_id (TestID, optional): Test ID to run. Not required if `unit_metrics` provided.
        params (dict, optional): Parameters to customize test behavior. See test details for available parameters.
        param_grid (Union[Dict[str, List[Any]], List[Dict[str, Any]]], optional): For comparison tests, either:
            - Dict mapping parameter names to lists of values (creates Cartesian product)
            - List of parameter dictionaries to test
        inputs (Dict[str, Any], optional): Test inputs (models/datasets initialized with vm.init_model/dataset)
        input_grid (Union[Dict[str, List[Any]], List[Dict[str, Any]]], optional): For comparison tests, either:
            - Dict mapping input names to lists of values (creates Cartesian product)
            - List of input dictionaries to test
        name (str, optional): Test name (required for composite metrics)
        unit_metrics (list, optional): Unit metric IDs to run as composite metric
        show (bool, optional): Whether to display results. Defaults to True.
        generate_description (bool, optional): Whether to generate a description. Defaults to True.

    Returns:
        TestResult: A TestResult object containing the test results

    Raises:
        ValueError: If the test inputs are invalid
        LoadTestError: If the test class fails to load
    """
    if not test_id and not (name and unit_metrics):
        raise ValueError(
            "`test_id` or both `name` and `unit_metrics` must be provided to run a test"
        )

    if bool(unit_metrics) != bool(name):
        raise ValueError("`name` and `unit_metrics` must be provided together")

    if input_grid and (kwargs or inputs):
        raise ValueError("Cannot provide `input_grid` along with `inputs`")

    if param_grid and params:
        raise ValueError("Cannot provide `param_grid` along with `params`")

    if unit_metrics:
        if not test_id:
            name = "".join(word.capitalize() for word in name.split())
            test_id = f"validmind.composite_metric.{name}"

        result = _run_composite_test(
            test_id=test_id,
            metric_ids=unit_metrics,
            inputs=inputs,
            params=params,
            show=show,
            generate_description=generate_description,
        )

    elif input_grid or param_grid:
        result = _run_comparison_test(
            test_id=test_id,
            inputs=inputs,
            input_grid=input_grid,
            param_grid=param_grid,
            name=name,
            unit_metrics=unit_metrics,
            params=params,
            show=show,
            generate_description=generate_description,
        )

    else:
        test_func = load_test(test_id)

        inputs = inputs or kwargs or {}
        params = params or {}

        input_kwargs, param_kwargs = _get_test_kwargs(test_func, inputs, params)

        raw_result = test_func(**input_kwargs, **param_kwargs)

        result = build_test_result(
            raw_result,
            test_id,
            input_kwargs,
            param_kwargs,
            getdoc(test_func),
            generate_description,
        )

    if show:
        result.show()

    return result
