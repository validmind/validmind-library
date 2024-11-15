# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from inspect import getdoc
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

from validmind.ai.test_descriptions import get_result_description
from validmind.errors import MissingRequiredTestInputError
from validmind.input_registry import input_registry
from validmind.logging import get_logger
from validmind.utils import test_id_to_name
from validmind.vm_models.input import VMInput
from validmind.vm_models.result import TestResult

from .__types__ import TestID
from ._comparison import combine_results, get_comparison_test_configs
from .load import _test_description, describe_test, load_test
from .output import process_output

logger = get_logger(__name__)


def _get_test_kwargs(
    test_func: callable, inputs: Dict[str, Any], params: Dict[str, Any]
):
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


def build_test_result(
    outputs: Union[Any, Tuple[Any, ...]],
    test_id: str,
    inputs: Dict[str, Union[VMInput, List[VMInput]]],
    params: Dict[str, Any],
    description: str,
    generate_description: bool = True,
):
    ref_id = str(uuid4())

    result = TestResult(
        result_id=test_id,
        ref_id=ref_id,
        inputs=inputs,
        params=params,
    )

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for item in outputs:
        process_output(item, result)

    result.description = get_result_description(
        test_id=test_id,
        test_description=description,
        tables=result.tables,
        figures=result.figures,
        metric=result.metric,
        should_generate=generate_description,
    )

    return result


def _run_composite_test(
    test_id: TestID,
    metric_ids: List[TestID],
    inputs: Dict[str, Union[VMInput, List[VMInput]]],
    params: Dict[str, Any],
    generate_description: bool = True,
):
    results = [
        run_test(
            test_id=metric_id,
            inputs=inputs,
            params=params,
            show=False,
            generate_description=False,
        )
        for metric_id in metric_ids
    ]

    # make sure to use is not None to handle for falsy values
    if not all(result.metric is not None for result in results):
        raise ValueError("All tests must return a metric when used as a composite test")

    return build_test_result(
        outputs=[
            {
                "Metric": test_id_to_name(result.result_id),
                "Value": result.metric,
            }
            for result in results
        ],  # pass in a single table with metric values as our 'outputs'
        test_id=test_id,
        inputs=results[0].inputs,
        params=results[0].params,
        description="\n\n".join(
            [_test_description(result.description, num_lines=1) for result in results]
        ),  # join truncated (first line only) test descriptions
        generate_description=generate_description,
    )


def _run_comparison_test(
    test_id: TestID,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    inputs: Union[Dict[str, Union[VMInput, List[VMInput]]], None] = None,
    params: Union[Dict[str, Any], None] = None,
    generate_description: bool = True,
):
    run_test_configs = get_comparison_test_configs(
        input_grid=input_grid,
        param_grid=param_grid,
        inputs=inputs,
        params=params,
    )

    results = [
        run_test(
            test_id=test_id,
            inputs=config["inputs"],
            params=config["params"],
            show=False,
            generate_description=False,
        )
        for config in run_test_configs
    ]

    combined_outputs, combined_inputs, combined_params = combine_results(results)

    return build_test_result(
        outputs=tuple(combined_outputs),
        test_id=test_id,
        inputs=combined_inputs,
        params=combined_params,
        description=describe_test(test_id, raw=True)[
            "Description"
        ],  # get description from test directly
        generate_description=generate_description,
    )


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
    # legacy support for passing inputs as kwargs
    inputs = inputs or kwargs

    if not test_id and not (name and unit_metrics):
        raise ValueError(
            "`test_id` or both `name` and `unit_metrics` must be provided to run a test"
        )

    if bool(unit_metrics) != bool(name):
        raise ValueError("`name` and `unit_metrics` must be provided together")

    if input_grid and inputs:
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
            generate_description=generate_description,
        )

    elif input_grid or param_grid:
        result = _run_comparison_test(
            test_id=test_id,
            inputs=inputs,
            input_grid=input_grid,
            params=params,
            param_grid=param_grid,
            generate_description=generate_description,
        )

    else:
        test_func = load_test(test_id)

        input_kwargs, param_kwargs = _get_test_kwargs(
            test_func, inputs or {}, params or {}
        )

        raw_result = test_func(**input_kwargs, **param_kwargs)

        result = build_test_result(
            outputs=raw_result,
            test_id=test_id,
            inputs=input_kwargs,
            params=param_kwargs,
            description=getdoc(test_func),
            generate_description=generate_description,
        )

    if show:
        result.show()

    return result
