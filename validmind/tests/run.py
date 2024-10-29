# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import itertools
from itertools import product
from typing import Any, Dict, List, Union
from uuid import uuid4

import pandas as pd

from validmind.ai.test_descriptions import get_description_metadata
from validmind.errors import LoadTestError
from validmind.logging import get_logger
from validmind.unit_metrics import run_metric
from validmind.unit_metrics.composite import load_composite_metric
from validmind.vm_models import (
    MetricResult,
    ResultSummary,
    ResultTable,
    ResultTableMetadata,
    TestContext,
    TestInput,
    ThresholdTestResults,
)
from validmind.vm_models.figure import is_matplotlib_figure, is_plotly_figure
from validmind.vm_models.test.result_wrapper import (
    MetricResultWrapper,
    ThresholdTestResultWrapper,
)

from .__types__ import TestID
from .load import load_test

logger = get_logger(__name__)


def _cartesian_product(input_grid: Dict[str, List[Any]]):
    """Get all possible combinations for a set of inputs"""
    return [dict(zip(input_grid, values)) for values in product(*input_grid.values())]


def _combine_summaries(summaries: List[Dict[str, Any]]):
    """Combine the summaries from multiple results

    Args:
        summaries (List[Dict[str, Any]]): A list of dictionaries where each dictionary
            has two keys: "inputs" and "summary". The "inputs" key should contain the
            inputs used for the test and the "summary" key should contain the actual
            summary object.

    Constraint: The summaries must all have the same structure meaning that each has
    the same number of tables in the same order with the same columns etc. This
    should always be the case for comparison tests since its the same test run
    multiple times with different inputs.
    """
    if not summaries[0]["summary"]:
        return None

    def combine_tables(table_index):
        combined_df = pd.DataFrame()

        for summary_obj in summaries:
            serialized = summary_obj["summary"].results[table_index].serialize()
            summary_df = pd.DataFrame(serialized["data"])
            summary_df = pd.concat(
                [
                    pd.DataFrame(summary_obj["inputs"], index=summary_df.index),
                    summary_df,
                ],
                axis=1,
            )
            combined_df = pd.concat([combined_df, summary_df], ignore_index=True)

        return ResultTable(
            data=combined_df.to_dict(orient="records"),
            metadata=summaries[0]["summary"].results[table_index].metadata,
        )

    return ResultSummary(
        results=[
            combine_tables(table_index)
            for table_index in range(len(summaries[0]["summary"].results))
        ]
    )


def _get_input_id(v):
    if isinstance(v, str):
        return v  # If v is a string, return it as is.
    elif isinstance(v, list) and all(hasattr(item, "input_id") for item in v):
        # If v is a list and all items have an input_id attribute, join their input_id values.
        return ", ".join(item.input_id for item in v)
    elif hasattr(v, "input_id"):
        return v.input_id  # If v has an input_id attribute, return it.
    return str(v)  # Otherwise, return the string representation of v.


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


def _combine_unit_metrics(results: List[MetricResultWrapper]):
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


def metric_comparison(
    results: List[MetricResultWrapper],
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

    merged_summary = _combine_summaries(
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

    return MetricResultWrapper(
        result_id=test_id,
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
        output_template=output_template,
        metric=MetricResult(
            key=test_id,
            ref_id=ref_id,
            value=[],
            summary=merged_summary,
        ),
        figures=merged_figures,
    )


def threshold_test_comparison(
    results: List[ThresholdTestResultWrapper],
    test_id: TestID,
    input_groups: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
    output_template: str = None,
    generate_description: bool = True,
):
    """Build a comparison result for multiple threshold test results"""
    ref_id = str(uuid4())

    input_group_strings = []

    for group in input_groups:
        new_group = {}
        for k, v in group.items():
            if isinstance(v, str):
                new_group[k] = v
            elif hasattr(v, "input_id"):
                new_group[k] = v.input_id
            elif isinstance(v, list) and all(hasattr(item, "input_id") for item in v):
                new_group[k] = ", ".join([item.input_id for item in v])
            else:
                raise ValueError(f"Unsupported type for value: {v}")
        input_group_strings.append(new_group)

    merged_summary = _combine_summaries(
        [
            {"inputs": input_group_strings[i], "summary": result.test_results.summary}
            for i, result in enumerate(results)
        ]
    )
    merged_figures = _combine_figures(
        [result.figures for result in results], input_groups
    )

    # Patch figure metadata so they are connected to the comparison result
    if merged_figures and len(merged_figures):
        for i, figure in enumerate(merged_figures):
            figure.key = f"{figure.key}-{i}"
            figure.metadata["_name"] = test_id
            figure.metadata["_ref_id"] = ref_id

    return ThresholdTestResultWrapper(
        result_id=test_id,
        result_metadata=[
            get_description_metadata(
                test_id=test_id,
                default_description=f"Comparison test result for {test_id}",
                summary=merged_summary.serialize() if merged_summary else None,
                figures=merged_figures,
                prefix="test_description",
                should_generate=generate_description,
            )
        ],
        inputs=[
            input if isinstance(input, str) else input.input_id
            for group in input_groups
            for input in group.values()
        ],
        output_template=output_template,
        test_results=ThresholdTestResults(
            test_name=test_id,
            ref_id=ref_id,
            # TODO: when we have param_grid support, this will need to be updated
            params=results[0].test_results.params,
            passed=all(result.test_results.passed for result in results),
            results=[],
            summary=merged_summary,
        ),
        figures=merged_figures,
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
    if isinstance(results[0], MetricResultWrapper):
        func = metric_comparison
    else:
        func = threshold_test_comparison

    result = func(
        results, test_id, inputs_params_product, output_template, generate_description
    )

    if show:
        result.show()

    return result


def run_test(
    test_id: TestID = None,
    params: Dict[str, Any] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]] = None,
    inputs: Dict[str, Any] = None,
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]] = None,
    name: str = None,
    unit_metrics: List[TestID] = None,
    output_template: str = None,
    show: bool = True,
    __generate_description: bool = True,
    **kwargs,
) -> Union[MetricResultWrapper, ThresholdTestResultWrapper]:
    """Run a test by test ID.
    test_id (TestID, optional): The test ID to run. Not required if `unit_metrics` is provided.
    params (dict, optional): A dictionary of parameters to pass into the test. Params
        are used to customize the test behavior and are specific to each test. See the
        test details for more information on the available parameters. Defaults to None.
    param_grid (Union[Dict[str, List[Any]], List[Dict[str, Any]]], optional): To run
        a comparison test, provide either a dictionary of parameters where the keys are
        the parameter names and the values are lists of different parameters, or a list of
        dictionaries where each dictionary is a set of parameters to run the test with.
        This will run the test multiple times with different sets of parameters and then
        combine the results into a single output. When passing a dictionary, the grid
        will be created by taking the Cartesian product of the parameter lists. Its simply
        a more convenient way of forming the param grid as opposed to passing a list of
        all possible combinations. Defaults to None.
    inputs (Dict[str, Any], optional): A dictionary of test inputs to pass into the
        test. Inputs are either models or datasets that have been initialized using
        vm.init_model() or vm.init_dataset(). Defaults to None.
    input_grid (Union[Dict[str, List[Any]], List[Dict[str, Any]]], optional): To run
        a comparison test, provide either a dictionary of inputs where the keys are
        the input names and the values are lists of different inputs, or a list of
        dictionaries where each dictionary is a set of inputs to run the test with.
        This will run the test multiple times with different sets of inputs and then
        combine the results into a single output. When passing a dictionary, the grid
        will be created by taking the Cartesian product of the input lists. Its simply
        a more convenient way of forming the input grid as opposed to passing a list of
        all possible combinations. Defaults to None.
    name (str, optional): The name of the test (used to create a composite metric
        out of multiple unit metrics) - required when running multiple unit metrics
    unit_metrics (list, optional): A list of unit metric IDs to run as a composite
        metric - required when running multiple unit metrics
    output_template (str, optional): A jinja2 html template to customize the output
        of the test. Defaults to None.
    show (bool, optional): Whether to display the results. Defaults to True.
    **kwargs: Keyword inputs to pass into the test (same as `inputs` but as keyword
        args instead of a dictionary):
        - dataset: A validmind Dataset object or a Pandas DataFrame
        - model: A model to use for the test
        - models: A list of models to use for the test
        - dataset: A validmind Dataset object or a Pandas DataFrame
    """

    # Validate input arguments with helper functions
    validate_test_inputs(test_id, name, unit_metrics)
    validate_grid_inputs(input_grid, kwargs, inputs, param_grid, params)

    # Handle composite metric creation
    if unit_metrics:
        test_id = generate_composite_test_id(name, test_id)

    # Run comparison tests if applicable
    if input_grid or param_grid:
        return run_comparison_test_with_grids(
            test_id,
            inputs,
            input_grid,
            param_grid,
            name,
            unit_metrics,
            params,
            output_template,
            show,
            __generate_description,
        )

    # Run unit metric tests
    if test_id.startswith("validmind.unit_metrics"):
        # TODO: as we move towards a more unified approach to metrics
        # we will want to make everything functional and remove the
        # separation between unit metrics and "normal" metrics
        return run_metric(test_id, inputs=inputs, params=params, show=show)

    # Load the appropriate test class
    TestClass = load_test_class(test_id, unit_metrics, name)

    # Create and run the test
    test = TestClass(
        test_id=test_id,
        context=TestContext(),
        inputs=TestInput({**kwargs, **(inputs or {})}),
        output_template=output_template,
        params=params,
        generate_description=__generate_description,
    )

    test.run()

    if show:
        test.result.show()

    return test.result


def validate_test_inputs(test_id, name, unit_metrics):
    """Validate the main test inputs for `test_id`, `name`, and `unit_metrics`."""
    if not test_id and not (name and unit_metrics):
        raise ValueError(
            "`test_id` or both `name` and `unit_metrics` must be provided to run a test"
        )

    if bool(unit_metrics) != bool(name):
        raise ValueError("`name` and `unit_metrics` must be provided together")


def validate_grid_inputs(input_grid, kwargs, inputs, param_grid, params):
    """Validate the grid inputs to avoid conflicting parameters."""
    if input_grid and (kwargs or inputs):
        raise ValueError("Cannot provide `input_grid` along with `inputs` or `kwargs`")

    if param_grid and (kwargs or params):
        raise ValueError("Cannot provide `param_grid` along with `params` or `kwargs`")


def generate_composite_test_id(name, test_id):
    """Generate a composite test ID if unit metrics are provided."""
    metric_id_name = "".join(word.capitalize() for word in name.split())
    return f"validmind.composite_metric.{metric_id_name}" or test_id


def run_comparison_test_with_grids(
    test_id,
    inputs,
    input_grid,
    param_grid,
    name,
    unit_metrics,
    params,
    output_template,
    show,
    generate_description,
):
    """Run a comparison test based on the presence of input and param grids."""
    if input_grid and param_grid:
        return run_comparison_test(
            test_id,
            input_grid,
            name=name,
            unit_metrics=unit_metrics,
            param_grid=param_grid,
            output_template=output_template,
            show=show,
            generate_description=generate_description,
        )
    if input_grid:
        return run_comparison_test(
            test_id,
            input_grid,
            name=name,
            unit_metrics=unit_metrics,
            params=params,
            output_template=output_template,
            show=show,
            generate_description=generate_description,
        )
    if param_grid:
        return run_comparison_test(
            test_id,
            inputs=inputs,
            name=name,
            unit_metrics=unit_metrics,
            param_grid=param_grid,
            output_template=output_template,
            show=show,
            generate_description=generate_description,
        )


def load_test_class(test_id, unit_metrics, name):
    """Load the appropriate test class based on `test_id` and unit metrics."""
    if unit_metrics:
        metric_id_name = "".join(word.capitalize() for word in name.split())
        error, TestClass = load_composite_metric(
            unit_metrics=unit_metrics, metric_name=metric_id_name
        )
        if error:
            raise LoadTestError(error)
        return TestClass
    return load_test(test_id, reload=True)
