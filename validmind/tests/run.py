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
from validmind.vm_models import VMDataset, VMInput
from validmind.vm_models.result import ResultTable, TestResult

from .__types__ import TestID
from .load import load_test
from .output import process_output

logger = get_logger(__name__)


def _check_for_sensitive_data(tables: List[ResultTable], inputs: Dict[str, VMInput]):
    """Check if a table contains raw data from input datasets"""
    dataset_columns = {
        col: len(input_obj.df)
        for input_obj in inputs.values()
        if isinstance(input_obj, VMDataset)
        for col in input_obj.columns
    }

    for i, table in enumerate(tables):
        table_columns = {col: len(table.data) for col in table.data.columns}

        offending_columns = [
            col
            for col in table_columns
            if col in dataset_columns and table_columns[col] == dataset_columns[col]
        ]

        if offending_columns:
            name = table.title or i
            raise ValueError(
                f"Raw input data found in table ({name}), pass `unsafe=True` "
                f"or remove the offending columns: {offending_columns}"
            )


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


def build_test_result(
    outputs: Union[Any, Tuple[Any, ...]],
    test_id: str,
    inputs: Dict[str, Union[VMInput, List[VMInput]]],
    params: Dict[str, Any],
    description: str = None,
    generate_description: bool = True,
):
    ref_id = str(uuid4())

    result = TestResult(
        result_id=test_id,
        ref_id=ref_id,
        inputs=[
            sub_i.input_id if hasattr(sub_i, "input_id") else sub_i
            for i in inputs
            for sub_i in (i if isinstance(i, list) else [i])
        ],
        params=params,
    )

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for item in outputs:
        process_output(item, result)

    _check_for_sensitive_data(result.tables, inputs)

    result.description = get_result_description(
        test_id=test_id,
        test_description=description,
        tables=result.tables,
        figures=result.figures,
        metric=result.metric,
        should_generate=generate_description,
    )

    return result


def run_composite_test(*args, **kwargs):
    raise NotImplementedError("Composite tests are not yet implemented")


def run_comparison_test(*args, **kwargs):
    raise NotImplementedError("Comparison tests are not yet implemented")


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
    # Validation
    if not test_id and not (name and unit_metrics):
        raise ValueError(
            "`test_id` or both `name` and `unit_metrics` must be provided to run a test"
        )

    if bool(unit_metrics) != bool(name):
        raise ValueError("`name` and `unit_metrics` must be provided together")

    if input_grid and (kwargs or inputs):
        raise ValueError("Cannot provide `input_grid` along with `inputs` or `kwargs`")

    if param_grid and (kwargs or params):
        raise ValueError("Cannot provide `param_grid` along with `params` or `kwargs`")

    if unit_metrics:
        if not test_id:
            name = "".join(word.capitalize() for word in name.split())
            test_id = f"validmind.composite_metric.{name}"

        return run_composite_test(
            test_id=test_id,
            unit_metrics=unit_metrics,
            inputs=inputs,
            params=params,
            show=show,
            generate_description=generate_description,
        )

    if input_grid or param_grid:
        return run_comparison_test(
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
