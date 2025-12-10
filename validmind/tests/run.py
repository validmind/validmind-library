# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import platform
import pprint
import subprocess
import time
from datetime import datetime
from inspect import getdoc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from validmind import __version__
from validmind.ai.test_descriptions import get_result_description
from validmind.errors import MissingRequiredTestInputError
from validmind.input_registry import input_registry
from validmind.logging import get_logger
from validmind.utils import test_id_to_name
from validmind.vm_models.input import VMInput
from validmind.vm_models.result import TestResult

from .__types__ import TestID
from .comparison import combine_results, get_comparison_test_configs
from .load import _test_description
from .output import process_output

logger = get_logger(__name__)


# shouldn't change once initialized
_run_metadata = {}


def _get_pip_freeze():
    """Get a dict of package names and versions"""
    output = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    parsed = {}

    for line in output.split("\n"):
        if not line:
            continue

        if "==" in line:
            package, version = line.split("==")
            parsed[package] = version
        elif " @ " in line:
            package = line.split(" @ ")[0]
            parsed[package] = "__editable__"

    return parsed


def _get_run_metadata(**metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Get metadata for a test run result"""
    if not _run_metadata:
        _run_metadata["validmind"] = {"version": __version__}
        _run_metadata["python"] = {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        }
        _run_metadata["platform"] = platform.platform()

        try:
            _run_metadata["pip"] = _get_pip_freeze()
        except Exception:
            pass

    return {
        **_run_metadata,
        **metadata,
        "timestamp": datetime.now().isoformat(),
    }


def _validate_context(
    context: Union[Dict[str, str], None],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Validate the context dictionary and return extracted values.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: test_description, instructions, additional_context
    """
    context = context or {}
    allowed_context_keys = {"test_description", "instructions", "additional_context"}

    # Validate keys
    invalid_keys = set(context.keys()) - allowed_context_keys
    if invalid_keys:
        raise ValueError(
            f"Invalid context keys: {invalid_keys}. "
            f"Allowed keys are: {allowed_context_keys}"
        )

    # Validate value types
    for key, value in context.items():
        if not isinstance(value, str):
            raise ValueError(f"Context value for key '{key}' must be a string.")

    return (
        context.get("test_description"),
        context.get("instructions"),
        context.get("additional_context"),
    )


def _get_test_kwargs(
    test_func: callable, inputs: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Insepect function signature to build kwargs to pass the inputs and params
    that the test function expects

    Args:
        test_func (callable): Test function to inspect
        inputs (dict): Test inputs... different formats are supported
            e.g. {"dataset": dataset, "model": "model_id"}
                 {"datasets": [dataset1, "dataset2_id"]}
                 {"datasets": ("dataset1_id", "dataset2_id")}
                 {"dataset": {
                     "input_id": "dataset2_id",
                     "options": {"columns": ["col1", "col2"]},
                 }}
        params (dict): Test parameters e.g. {"param1": 1, "param2": 2}

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Tuple of input and param kwargs
    """
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
            try:
                _input = input_registry.get(key=_input["input_id"]).with_options(
                    **{k: v for k, v in _input.items() if k != "input_id"}
                )
            except KeyError as e:
                raise ValueError(
                    "Input dictionary must contain an 'input_id' key "
                    "to retrieve the input object from the input registry."
                ) from e

        input_kwargs[key] = _input

    param_kwargs = {
        key: value for key, value in params.items() if key in test_func.params
    }

    return input_kwargs, param_kwargs


def build_test_result(
    outputs: Union[Any, Tuple[Any, ...]],
    test_id: str,
    test_doc: str,
    inputs: Dict[str, Union[VMInput, List[VMInput]]],
    params: Union[Dict[str, Any], None],
    title: Optional[str] = None,
    test_func: Optional[Callable] = None,
):
    """Build a TestResult object from a set of raw test function outputs"""
    ref_id = str(uuid4())

    result = TestResult(
        result_id=test_id,
        title=title,
        ref_id=ref_id,
        inputs=inputs,
        params=params if params else None,  # None if empty dict or None
        doc=test_doc,
        _is_scorer_result=test_func is not None
        and hasattr(test_func, "_is_scorer")
        and test_func._is_scorer,
    )

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for item in outputs:
        process_output(item, result, test_func)

    return result


def _run_composite_test(
    test_id: TestID,
    metric_ids: List[TestID],
    inputs: Union[Dict[str, Any], None],
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None],
    params: Union[Dict[str, Any], None],
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None],
    title: Optional[str] = None,
):
    """Run a composite test i.e. a test made up of multiple metrics"""
    # no-op: _test_description imported at module scope now that circular import is resolved

    results = [
        run_test(
            test_id=metric_id,
            inputs=inputs,
            input_grid=input_grid,
            params=params,
            param_grid=param_grid,
            show=False,
            generate_description=False,
            title=title,
        )
        for metric_id in metric_ids
    ]

    # make sure to use is not None to handle for falsy values
    if not all(result.metric is not None for result in results):
        raise ValueError("All tests must return a metric when used as a composite test")

    # Create composite docstring from all test results
    composite_doc = "\n\n".join(
        [
            f"{test_id_to_name(result.result_id)}:\n{_test_description(result.doc)}"
            for result in results
        ]
    )

    return build_test_result(
        outputs=[
            {
                "Metric": test_id_to_name(result.result_id),
                "Value": result.metric,
            }
            for result in results
        ],  # pass in a single table with metric values as our 'outputs'
        test_id=test_id,
        test_doc=composite_doc,
        inputs=results[0].inputs,
        params=results[0].params,
        title=title,
    )


def _run_comparison_test(
    test_id: Union[TestID, None],
    name: Union[str, None],
    unit_metrics: Union[List[TestID], None],
    inputs: Union[Dict[str, Any], None],
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None],
    params: Union[Dict[str, Any], None],
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None],
    title: Optional[str] = None,
    show_params: bool = True,
):
    """Run a comparison test i.e. a test that compares multiple outputs of a test across
    different input and/or param combinations"""
    from .load import describe_test

    run_test_configs = get_comparison_test_configs(
        input_grid=input_grid,
        param_grid=param_grid,
        inputs=inputs,
        params=params,
    )

    results = [
        run_test(
            test_id=test_id,
            name=name,
            unit_metrics=unit_metrics,
            inputs=config["inputs"],
            params=config["params"],
            show=False,
            generate_description=False,
            title=title,
            show_params=show_params,
        )
        for config in run_test_configs
    ]

    # composite tests have a test_id thats built from the name
    if not test_id:
        test_id = results[0].result_id
        test_doc = results[0].doc
    else:
        test_doc = describe_test(test_id, raw=True)["Description"]

    combined_outputs, combined_inputs, combined_params = combine_results(
        results, show_params
    )

    return build_test_result(
        outputs=combined_outputs,
        test_id=test_id,
        test_doc=test_doc,
        inputs=combined_inputs,
        params=combined_params,
        title=title,
    )


def _run_test(
    test_id: TestID,
    inputs: Dict[str, Any],
    params: Dict[str, Any],
    title: Optional[str] = None,
    doc: Optional[str] = None,
):
    """Run a standard test and return a TestResult object"""
    from .load import load_test

    test_func = load_test(test_id)
    input_kwargs, param_kwargs = _get_test_kwargs(
        test_func=test_func,
        inputs=inputs or {},
        params=params or {},
    )

    raw_result = test_func(**input_kwargs, **param_kwargs)

    # Use custom doc if provided, otherwise use the test function's docstring
    _doc = doc if doc is not None else getdoc(test_func)

    return build_test_result(
        outputs=raw_result,
        test_id=test_id,
        test_doc=_doc,
        inputs=input_kwargs,
        params=param_kwargs,
        title=title,
        test_func=test_func,
    )


def run_test(  # noqa: C901
    test_id: Union[TestID, None] = None,
    name: Union[str, None] = None,
    unit_metrics: Union[List[TestID], None] = None,
    inputs: Union[Dict[str, Any], None] = None,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    params: Union[Dict[str, Any], None] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    show: bool = True,
    generate_description: bool = True,
    title: Optional[str] = None,
    post_process_fn: Union[Callable[[TestResult], None], None] = None,
    show_params: bool = True,
    context: Union[Dict[str, str], None] = None,
    **kwargs,
) -> TestResult:
    """Run a ValidMind or custom test

    This function is the main entry point for running tests. It can run simple unit metrics,
    ValidMind and custom tests, composite tests made up of multiple unit metrics and comparison
    tests made up of multiple tests.

    Args:
        test_id (TestID, optional): Test ID to run. Not required if `name` and `unit_metrics` provided.
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
        title (str, optional): Custom title for the test result
        post_process_fn (Callable[[TestResult], None], optional): Function to post-process the test result
        show_params (bool, optional): Whether to include parameter values in figure titles for comparison tests. Defaults to True.
        context (Dict[str, str], optional): Context for test description generation. Supported keys:
            - 'test_description': Custom docstring to override the test's built-in documentation
            - 'instructions': Instructions for the LLM to format the description output
            - 'additional_context': Background information for the LLM to contextualize results

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
            "`test_id` or `name` and `unit_metrics` must be provided to run a test"
        )

    if bool(unit_metrics) != bool(name):
        raise ValueError("`name` and `unit_metrics` must be provided together")

    if input_grid and inputs:
        raise ValueError("Cannot provide `input_grid` along with `inputs`")

    if param_grid and params:
        raise ValueError("Cannot provide `param_grid` along with `params`")

    # Validate and extract individual context values
    test_description, instructions, additional_context = _validate_context(context)

    start_time = time.perf_counter()

    if input_grid or param_grid:
        result = _run_comparison_test(
            test_id=test_id,
            title=title,
            name=name,
            unit_metrics=unit_metrics,
            inputs=inputs,
            input_grid=input_grid,
            params=params,
            param_grid=param_grid,
            show_params=show_params,
        )

    elif unit_metrics:
        name = "".join(word.capitalize() for word in name.split())
        test_id = f"validmind.composite_metric.{name}"

        result = _run_composite_test(
            test_id=test_id,
            metric_ids=unit_metrics,
            inputs=inputs,
            input_grid=input_grid,
            params=params,
            param_grid=param_grid,
            title=title,
        )

    else:
        result = _run_test(test_id, inputs, params, title, test_description)

    end_time = time.perf_counter()
    result.metadata = _get_run_metadata(duration_seconds=end_time - start_time)

    if post_process_fn:
        result = post_process_fn(result)

    if not result.description:
        result.description = get_result_description(
            test_id=test_id,
            test_description=result.doc,
            tables=result.tables,
            figures=result.figures,
            metric=result.metric,
            should_generate=generate_description,
            title=title,
            instructions=instructions,
            additional_context=additional_context,
            params=result.params,
        )

    if show:
        result.show()

    return result


def print_env():
    """Prints a log of the running environment for debugging.

    Output includes: ValidMind Library version, operating system details, installed dependencies, and the ISO 8601 timestamp at log creation.
    """
    e = _get_run_metadata()
    pprint.pp(e)
