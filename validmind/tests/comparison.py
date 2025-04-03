# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from itertools import product
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from validmind.logging import get_logger
from validmind.utils import test_id_to_name
from validmind.vm_models.figure import (
    is_matplotlib_figure,
    is_plotly_figure,
    is_png_image,
)
from validmind.vm_models.input import VMInput
from validmind.vm_models.result import RawData, ResultTable, TestResult

logger = get_logger(__name__)


def _cartesian_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations for a grid of inputs or parameters.

    Args:
        grid: A dictionary where each key corresponds to a parameter name and the associated list contains possible values.

    Returns:
        A list of dictionaries representing all combinations of the parameter values.

    Example:
        _cartesian_product({"a": [1, 2], "b": [3, 4]})
        >>> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    if not grid:
        return [{}]
    return [dict(zip(grid.keys(), values)) for values in product(*grid.values())]


def _get_input_key(input_obj_or_list: Union[VMInput, List[VMInput]]) -> str:
    """Create a key for a given input or list of inputs"""
    if isinstance(input_obj_or_list, list):
        return ",".join(item.input_id for item in input_obj_or_list)

    return input_obj_or_list.input_id


def _get_unique_inputs(results: List[TestResult]) -> Dict[str, set]:
    """Get only the inputs that are not the same across all results"""
    unique_inputs = {}

    for result in results:
        if not result.inputs:
            continue

        for func_input_name, input_obj_or_list in result.inputs.items():
            if isinstance(input_obj_or_list, list):
                key = ",".join(_get_input_key(item) for item in input_obj_or_list)
            else:
                key = _get_input_key(input_obj_or_list)

            unique_inputs.setdefault(func_input_name, set()).add(key)

    return unique_inputs


def _get_unique_params(results: List[TestResult]) -> Dict[str, List[Any]]:
    """Get only the params that are not the same across all results"""
    param_values = {}
    for result in results:
        if not result.params:
            continue

        for name, value in result.params.items():
            param_values.setdefault(name, []).append(value)

    unique_params = {}
    for name, values in param_values.items():
        unique_values = []
        for value in values:
            if not any(value == x for x in unique_values):
                unique_values.append(value)

        unique_params[name] = unique_values

    return unique_params


def _get_table_metadata(
    result: TestResult, results: List[TestResult]
) -> Dict[str, Any]:
    """Create a metadata dict with unique inputs and params for a table row"""
    metadata = {}
    unique_inputs = _get_unique_inputs(results)
    unique_params = _get_unique_params(results)

    if result.inputs:
        for input_name, input_obj in result.inputs.items():
            if len(unique_inputs[input_name]) > 1:
                metadata[input_name] = _get_input_key(input_obj)

    if result.params:
        for param_name, param_value in result.params.items():
            if len(unique_params[param_name]) > 1:
                metadata[param_name] = param_value

    return metadata


def _combine_single_table(results: List[TestResult], table_index: int) -> pd.DataFrame:
    """
    Combine a single table across multiple test results.

    Args:
        results: A list of TestResult objects.
        table_index: The index of the table to combine.

    Returns:
        A pandas DataFrame combining the tables with added metadata columns.
    """
    combined_tables = []

    for result in results:
        metadata = _get_table_metadata(result, results)
        table_data = result.tables[table_index].data

        if metadata:
            metadata_df = pd.DataFrame([metadata] * len(table_data))
            table_data = pd.concat([metadata_df, table_data], axis=1)

        combined_tables.append(table_data)

    return pd.concat(combined_tables, ignore_index=True)


def _combine_tables(results: List[TestResult]) -> List[pd.DataFrame]:
    """Combine tables from multiple test results

    # TODO: retain table titles
    """
    if not results[0].tables:
        return []

    return [_combine_single_table(results, i) for i in range(len(results[0].tables))]


def _build_input_param_string(
    result: TestResult, results: List[TestResult], show_params: bool
) -> str:
    """Build a string repr of unique inputs + params for a figure title"""
    parts = []
    unique_inputs = _get_unique_inputs(results)

    # if theres only one unique value for an input or param, don't show it
    # however, if there is only one unique value for all inputs then show it
    if result.inputs:
        should_show = all(
            len(unique_inputs[input_name]) == 1 for input_name in unique_inputs
        )
        for input_name, input_obj in result.inputs.items():
            if should_show or len(unique_inputs[input_name]) > 1:
                input_val = _get_input_key(input_obj)
                parts.append(f"{input_name}={input_val}")

    # Handle params if show_params is enabled
    if show_params and result.params:
        unique_params = _get_unique_params(results)
        # If there's only one unique value for a param, don't show it
        # unless there is only one unique value for all params and no inputs shown
        should_show = (
            all(len(unique_params[param_name]) == 1 for param_name in unique_params)
            and not parts
        )
        for param_name, param_value in result.params.items():
            if should_show or len(unique_params[param_name]) > 1:
                # Convert the param_value to a string representation
                if isinstance(param_value, list):
                    # For lists, join elements with commas
                    str_value = ",".join(str(v) for v in param_value)
                elif hasattr(param_value, "__str__"):
                    # Use string representation if available
                    str_value = str(param_value)
                else:
                    # Default fallback
                    str_value = repr(param_value)

                parts.append(f"{param_name}={str_value}")

    return ", ".join(parts)


def _update_figure_title(figure: Any, input_param_str: str) -> None:
    """
    Update the title of a figure with input and parameter information.

    Args:
        figure: A figure object (matplotlib, plotly, or PNG image).
        input_param_str: A string of input and parameter information.

    Raises:
        ValueError: If the figure type is unsupported.
    """
    if not input_param_str:
        return

    new_title = f"{{curr_title}} (for {input_param_str})"

    if is_matplotlib_figure(figure):
        curr_title = figure._suptitle.get_text() if figure._suptitle else ""
        figure.suptitle(new_title.format(curr_title=curr_title))
    elif is_plotly_figure(figure):
        curr_title = figure.layout.title.text
        figure.layout.title.text = new_title.format(curr_title=curr_title)
    elif is_png_image(figure):
        logger.warning("Unable to update title for PNG image figure.")
    else:
        raise ValueError(f"Unsupported figure type: {type(figure)}")


def _combine_figures(results: List[TestResult], show_params: bool) -> List[Any]:
    """Combine figures from multiple test results (gets raw figure objects, not vm Figures)"""
    combined_figures = []

    for result in results:
        for figure in result.figures or []:
            # update the figure object in-place with the new title
            _update_figure_title(
                figure=figure.figure,
                input_param_str=_build_input_param_string(result, results, show_params),
            )
            combined_figures.append(figure)

    return combined_figures


def _handle_metrics(results: List[TestResult]) -> List[Any]:
    """Combine metrics from multiple test results"""
    # add a table with the metric value so it is combined into a single table
    for result in results:
        if result.metric:
            result.add_table(
                ResultTable(
                    data=[
                        {
                            "Metric": test_id_to_name(result.result_id),
                            "Value": result.metric,
                        }
                    ],
                    title=None,
                )
            )


def _combine_dict_values(items_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Combine values for each key in a dictionary, keeping only unique values"""
    combined = {}

    for name, value in items_dict.items():
        values = value if isinstance(value, list) else [value]

        unique_values = []
        for v in values:
            if not any(v == x for x in unique_values):
                unique_values.append(v)

        combined[name] = unique_values[0] if len(unique_values) == 1 else unique_values

    return combined


def get_comparison_test_configs(
    input_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]], None] = None,
    inputs: Union[Dict[str, Union[VMInput, List[VMInput]]], None] = None,
    params: Union[Dict[str, Any], None] = None,
) -> List[Dict[str, Any]]:
    """
    Generate test configurations based on input and parameter grids.

    Function inputs should be validated before calling this.

    Args:
        input_grid: A dictionary or list defining the grid of inputs.
        param_grid: A dictionary or list defining the grid of parameters.
        inputs: A dictionary of inputs.
        params: A dictionary of parameters.

    Returns:
        A list of test configurations.
    """

    # Convert list of dicts to dict of lists if necessary for input_grid
    def list_to_dict(grid_list):
        return {k: [d[k] for d in grid_list] for k in grid_list[0].keys()}

    # Handle input_grid the same way as before
    if isinstance(input_grid, list):
        input_grid = list_to_dict(input_grid)

    test_configs = []

    # Check if param_grid is a list of dictionaries
    is_param_grid_list = isinstance(param_grid, list)

    # Special handling for list-based param_grid
    if is_param_grid_list:
        if input_grid:
            # Generate all combinations of input_grid and each param dictionary
            input_combinations = _cartesian_product(input_grid)
            test_configs = [
                {"inputs": i, "params": p}
                for i in input_combinations
                for p in param_grid
            ]
        else:
            # Each dictionary in param_grid is a specific test configuration
            test_configs = [{"inputs": inputs or {}, "params": p} for p in param_grid]

    # Dictionary-based param_grid
    elif param_grid:
        if input_grid:
            input_combinations = _cartesian_product(input_grid)
            param_combinations = _cartesian_product(param_grid)
            test_configs = [
                {"inputs": i, "params": p}
                for i, p in product(input_combinations, param_combinations)
            ]
        else:
            param_combinations = _cartesian_product(param_grid)
            test_configs = [
                {"inputs": inputs or {}, "params": p} for p in param_combinations
            ]
    # Just input_grid, no param_grid
    elif input_grid:
        input_combinations = _cartesian_product(input_grid)
        test_configs = [
            {"inputs": i, "params": params or {}} for i in input_combinations
        ]

    return test_configs


def _combine_raw_data(results: List[TestResult]) -> RawData:
    """Combine RawData objects"""
    attribute_names = results[0].raw_data.__dict__.keys()

    # check that all the raw data objects have the same attributes
    for result in results:
        if not isinstance(result.raw_data, RawData):
            raise ValueError("All raw data objects must be of type RawData")
        if result.raw_data.__dict__.keys() != attribute_names:
            raise ValueError("RawData objects must have the same attributes")

    return RawData(
        **{
            key: [getattr(result.raw_data, key) for result in results]
            for key in attribute_names
        }
    )


def combine_results(
    results: List[TestResult],
    show_params: bool,
) -> Tuple[List[Any], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """
    Combine multiple test results into a single set of outputs.

    Args:
        results: A list of TestResult objects to combine.
        show_params: Whether to show parameter values in figure titles.

    Returns:
        A tuple containing:
            - A list of combined outputs (tables and figures).
            - A dictionary of inputs with lists of all values.
            - A dictionary of parameters with lists of all values.
    """
    # metrics are added as a table to each result so later they can be combined
    _handle_metrics(results)

    combined_outputs = []
    # handle tables (if any)
    combined_outputs.extend(_combine_tables(results))
    # handle figures (if any)
    combined_outputs.extend(_combine_figures(results, show_params))
    # handle threshold tests (i.e. tests that have pass/fail bool status)
    if results[0].passed is not None:
        combined_outputs.append(all(result.passed for result in results))
    # handle raw data (if any)
    if results[0].raw_data:
        combined_outputs.append(_combine_raw_data(results))

    # combine inputs and params
    combined_inputs = {}
    combined_params = {}

    for result in results:
        if result.inputs:
            for input_name, input_obj_or_list in result.inputs.items():
                combined_inputs.setdefault(input_name, []).extend(
                    input_obj_or_list
                    if isinstance(input_obj_or_list, list)
                    else [input_obj_or_list]
                )

        if result.params:
            for param_name, param_value in result.params.items():
                combined_params.setdefault(param_name, []).append(param_value)

    combined_inputs = _combine_dict_values(combined_inputs)
    combined_params = _combine_dict_values(combined_params)

    return (
        tuple(combined_outputs),
        combined_inputs,
        combined_params,
    )
