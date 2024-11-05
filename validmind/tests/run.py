# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import inspect
import itertools
from itertools import product
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import pandas as pd

from validmind.ai.test_descriptions import get_result_description
from validmind.errors import LoadTestError, MissingRequiredTestInputError
from validmind.input_registry import input_registry
from validmind.logging import get_logger
from validmind.unit_metrics import run_metric
from validmind.unit_metrics.composite import load_composite_metric
from validmind.vm_models import VMDataset, VMModel
from validmind.vm_models.figure import is_matplotlib_figure, is_plotly_figure
from validmind.vm_models.result import ResultTable, ResultTableMetadata, TestResult

from .__types__ import TestID
from .load import load_test

logger = get_logger(__name__)


INPUT_TYPE_MAP = {
    "dataset": VMDataset,
    "datasets": List[VMDataset],
    "model": VMModel,
    "models": List[VMModel],
}


def _cartesian_product(grid: Dict[str, List[Any]]):
    """Get all possible combinations for a set of inputs or params"""
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def _get_input_id(v):
    if isinstance(v, str):
        return v  # If v is a string, return it as is.
    elif isinstance(v, list) and all(hasattr(item, "input_id") for item in v):
        # If v is a list and all items have an input_id attribute, join their input_id values.
        return ", ".join(item.input_id for item in v)
    elif hasattr(v, "input_id"):
        return v.input_id  # If v has an input_id attribute, return it.
    return str(v)  # Otherwise, return the string representation of v.


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


def _combine_tables(
    tables_lists_with_inputs: List[Tuple[List[ResultTable], Dict[str, Any]]]
):
    """Combine the tables from multiple results

    Args:
        tables_lists_with_inputs (List[Tuple[List[ResultTable], Dict[str, Any]]]): A list
            of tuples where the first element is a list of ResultTable objects and the
            second element is a dictionary of inputs passed to the test that generated the
            tables.
    """
    # use the first set of tables as the primary since all should have the same schema/titles
    primary_tables = tables_lists_with_inputs[0][0]

    combined_tables = []

    # process first table in each list first then second, etc.
    for table_idx in range(len(primary_tables)):
        dfs = []

        for tables, inputs in tables_lists_with_inputs:
            table = tables[table_idx]

            for input_name, input_value in inputs.items():
                table.data[input_name] = _get_input_id(input_value)

            dfs.append(table.data)

        # combine into new table with all the rows and same title
        combined_tables.append(
            ResultTable(
                data=pd.concat(dfs, ignore_index=True),
                title=primary_tables[table_idx].title,
            )
        )

    return combined_tables


def _update_plotly_titles(figures, input_group, title_template):
    for figure in figures:

        current_title = figure.figure.layout.title.text

        input_description = " and ".join(
            f"{key}: {_get_input_id(value)}" for key, value in input_group.items()
        )

        figure.figure.layout.title.text = title_template.format(
            current_title=f"{current_title} " if current_title else "",
            input_description=input_description,
        )


def _update_matplotlib_titles(figures, input_group, title_template):
    for figure in figures:

        current_title = (
            figure.figure._suptitle.get_text() if figure.figure._suptitle else ""
        )

        input_description = " and ".join(
            f"{key}: {_get_input_id(value)}" for key, value in input_group.items()
        )

        figure.figure.suptitle(
            title_template.format(
                current_title=f"{current_title} " if current_title else "",
                input_description=input_description,
            )
        )


def _combine_figures(figure_lists: List[List[Any]], input_groups: List[Dict[str, Any]]):
    """Combine the figures from multiple results"""
    if not figure_lists[0]:
        return None

    title_template = "{current_title}({input_description})"

    for idx, figures in enumerate(figure_lists):
        input_group = input_groups[idx]["inputs"]
        if is_plotly_figure(figures[0].figure):
            _update_plotly_titles(figures, input_group, title_template)
        elif is_matplotlib_figure(figures[0].figure):
            _update_matplotlib_titles(figures, input_group, title_template)
        else:
            logger.warning("Cannot properly annotate png figures")

    return [figure for figures in figure_lists for figure in figures]


def _combine_unit_metrics(results: List[TestResult]):
    if not results[0].scalar:
        return

    for result in results:
        table = ResultTable(
            data=[{"value": result.scalar}],
            metadata=ResultTableMetadata(title="Unit Metrics"),
        )
        if not result.metric:
            result.metric = MetricResult(
                ref_id="will_be_overwritten",
                key=result.result_id,
                value=result.scalar,
                summary=ResultSummary(results=[table]),
            )
        else:
            result.metric.summary.results.append(table)


def build_comparison_result(
    results: List[TestResult],
    test_id: TestID,
    input_params_groups: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
    output_template: str = None,
    generate_description: bool = True,
):
    """Build a comparison result for multiple metric results"""
    ref_id = str(uuid4())

    # Treat param_groups and input_groups as empty lists if they are None or empty
    input_params_groups = input_params_groups or [{}]

    input_group_strings = []

    for input_params in input_params_groups:
        new_group = {}
        for param_k, param_v in input_params["params"].items():
            new_group[param_k] = param_v
        for metric_k, metric_v in input_params["inputs"].items():
            # Process values in the input group
            if isinstance(metric_v, str):
                new_group[metric_k] = metric_v
            elif hasattr(metric_v, "input_id"):
                new_group[metric_k] = metric_v.input_id
            elif isinstance(metric_v, list) and all(
                hasattr(item, "input_id") for item in metric_v
            ):
                new_group[metric_k] = ", ".join([item.input_id for item in metric_v])
            else:
                raise ValueError(f"Unsupported type for value: {metric_v}")
        input_group_strings.append(new_group)

    # handle unit metrics (scalar values) by adding it to the summary
    _combine_unit_metrics(results)

    merged_summary = _combine_tables(
        [
            {"inputs": input_group_strings[i], "summary": result.metric.summary}
            for i, result in enumerate(results)
        ]
    )
    merged_figures = _combine_figures(
        [result.figures for result in results], input_params_groups
    )

    # Patch figure metadata so they are connected to the comparison result
    if merged_figures and len(merged_figures):
        for i, figure in enumerate(merged_figures):
            figure.key = f"{figure.key}-{i}"
            figure.metadata["_name"] = test_id
            figure.metadata["_ref_id"] = ref_id

    return TestResult(
        result_id=test_id,
        result_description=get_result_description(
            test_id=test_id,
            test_description=results[0].result_description,
            tables=merged_tables,
            figures=merged_figures,
            passed=passed,
            should_generate=generate_description,
        ),
        result_metadata=[
            get_description_metadata(
                test_id=test_id,
                default_description=f"Comparison test result for {test_id}",
                summary=merged_summary.serialize() if merged_summary else None,
                figures=merged_figures,
                should_generate=generate_description,
            ),
        ],
        inputs=[
            item.input_id if hasattr(item, "input_id") else item
            for group in input_params_groups
            for input in group["inputs"].values()
            for item in (input if isinstance(input, list) else [input])
            if hasattr(item, "input_id") or isinstance(item, str)
        ],
        tables=merged_tables,
        figures=merged_figures,
        output_template=output_template,
    )


def run_comparison_test(
    test_id: TestID,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]] = None,
    inputs: Dict[str, Any] = None,
    name: str = None,
    unit_metrics: List[TestID] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]] = None,
    params: Dict[str, Any] = None,
    show: bool = True,
    output_template: str = None,
    generate_description: bool = True,
):
    """Run a comparison test"""
    if input_grid:
        if isinstance(input_grid, dict):
            input_groups = _cartesian_product(input_grid)
        else:
            input_groups = input_grid
    else:
        input_groups = list(inputs) if inputs else []

    if param_grid:
        if isinstance(param_grid, dict):
            param_groups = _cartesian_product(param_grid)
        else:
            param_groups = param_grid
    else:
        param_groups = list(params) if inputs else []

    input_groups = input_groups or [{}]
    param_groups = param_groups or [{}]
    # Use itertools.product to compute the Cartesian product
    inputs_params_product = [
        {
            "inputs": item1,
            "params": item2,
        }  # Merge dictionaries from input_groups and param_groups
        for item1, item2 in itertools.product(input_groups, param_groups)
    ]
    results = [
        run_test(
            test_id,
            name=name,
            unit_metrics=unit_metrics,
            inputs=inputs_params["inputs"],
            show=False,
            params=inputs_params["params"],
            __generate_description=False,
        )
        for inputs_params in (inputs_params_product or [{}])
    ]

    result = build_comparison_result(
        results, test_id, inputs_params_product, output_template, generate_description
    )

    if show:
        result.show()

    return result


def _inspect_signature(test_func: callable):
    inputs = {}
    params = {}

    for name, arg in inspect.signature(test_func).parameters.items():
        if name in INPUT_TYPE_MAP:
            inputs[name] = {"type": INPUT_TYPE_MAP[name]}
        else:
            params[name] = {
                "type": arg.annotation,
                "default": (
                    arg.default if arg.default is not inspect.Parameter.empty else None
                ),
            }

    return inputs, params


def _get_test_kwargs(test_func, inputs, params):
    input_kwargs = {}  # map function inputs (`dataset` etc) to actual objects

    func_inputs, func_params = _inspect_signature(test_func)

    for key in func_inputs.keys():
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
        key: params.get(key, func_params[key]["default"]) for key in func_params.keys()
    }

    return input_kwargs, param_kwargs


def _build_result(
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

    return TestResult(
        result_id=test_id,
        result_description=get_result_description(
            test_id=test_id,
            test_description=description,
            tables=tables,
            figures=figures,
            passed=passed,
            should_generate=generate_description,
        ),
        metric=scalars[0] if scalars else None,
        tables=tables,
        figures=figures,
        inputs=metric_inputs,
        params=params,
        output_template=output_template,
    )


def run_test(
    test_id: Union[TestID, None] = None,
    params: Union[Dict[str, Any], None] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    inputs: Union[Dict[str, Any], None] = None,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    name: Union[str, None] = None,
    unit_metrics: Union[List[TestID], None] = None,
    output_template: Union[str, None] = None,
    show: bool = True,
    __generate_description: bool = True,  # TODO: deprecate
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
        output_template (str, optional): Custom jinja2 HTML template for output
        show (bool, optional): Whether to display results. Defaults to True.
        generate_description (bool, optional): Whether to generate a description. Defaults to True.
        **kwargs: Additional test inputs:
            - dataset: ValidMind Dataset or DataFrame
            - model: Model to test
            - models: List of models to test

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

    generate_description = generate_description or __generate_description

    if unit_metrics:
        if not test_id:
            name = "".join(word.capitalize() for word in name.split())
            test_id = f"validmind.composite_metric.{name}"

        return run_composite_metric(
            test_id=test_id,
            unit_metrics=unit_metrics,
            inputs=inputs,
            params=params,
            output_template=output_template,
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
            output_template=output_template,
            show=show,
            generate_description=generate_description,
        )

    # Run unit metric tests
    if test_id.startswith("validmind.unit_metrics"):
        return run_metric(test_id, inputs=inputs, params=params, show=show)

    if unit_metrics:
        metric_id_name = "".join(word.capitalize() for word in name.split())
        error, TestClass = load_composite_metric(
            unit_metrics=unit_metrics, metric_name=metric_id_name
        )
        if error:
            raise LoadTestError(error)
        test = TestClass

    test_func = load_test(test_id, reload=True)

    # Create and run the test
    input_kwargs, param_kwargs = _get_test_kwargs(test_func, inputs, params)
    raw_result = test_func(**input_kwargs, **param_kwargs)
    result = build_test_result(
        raw_result, test_id, output_template, generate_description
    )

    if show:
        result.show()

    return result
