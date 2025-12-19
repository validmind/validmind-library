# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Module for listing and loading tests."""

import inspect
import json
from pprint import pformat
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)
from uuid import uuid4

import pandas as pd

from ..errors import LoadTestError, MissingDependencyError
from ..html_templates.content_blocks import test_content_block_html
from ..logging import get_logger
from ..utils import display, format_dataframe, fuzzy_match, md_to_html, test_id_to_name
from ..vm_models.dataset.dataset import VMDataset
from ..vm_models.figure import Figure
from ..vm_models.model import VMModel
from ..vm_models.result import ResultTable
from .__types__ import TestID
from ._store import test_provider_store, test_store

logger = get_logger(__name__)


try:
    from matplotlib.figure import Figure as MatplotlibFigure
except ImportError:
    MatplotlibFigure = None

try:
    from plotly.graph_objects import Figure as PlotlyFigure
except ImportError:
    PlotlyFigure = None

FIGURE_TYPES = tuple(
    item for item in (Figure, MatplotlibFigure, PlotlyFigure) if inspect.isclass(item)
)
TABLE_TYPES = (pd.DataFrame, ResultTable)
GENERIC_TABLE_TYPES = (list, dict)


INPUT_TYPE_MAP = {
    "dataset": VMDataset,
    "datasets": List[VMDataset],
    "model": VMModel,
    "models": List[VMModel],
}


def _inspect_return_type(annotation: Any) -> Tuple[bool, bool]:
    """
    Inspects a return type annotation to determine if it contains a Figure or Table.

    Returns a tuple (has_figure, has_table).
    """
    has_figure = False
    has_table = False

    origin = get_origin(annotation)
    args = get_args(annotation)

    # A Union means the return type could be one of several types.
    # A tuple in a type hint means multiple return values.
    # We recursively inspect the arguments of Union and tuple.
    if origin is Union or origin is tuple:
        for arg in args:
            fig, table = _inspect_return_type(arg)
            has_figure |= fig
            has_table |= table
        return has_figure, has_table

    check_type = origin if origin is not None else annotation

    if not inspect.isclass(check_type):
        return has_figure, has_table  # Can't do issubclass on non-class like Any

    if FIGURE_TYPES and issubclass(check_type, FIGURE_TYPES):
        has_figure = True

    if TABLE_TYPES and issubclass(check_type, TABLE_TYPES):
        has_table = True

    if check_type in GENERIC_TABLE_TYPES:
        has_table = True

    return has_figure, has_table


def _inspect_signature(
    test_func: Callable[..., Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Inspect a test function's signature to get inputs and parameters"""
    inputs = {}
    params = {}

    for name, arg in inspect.signature(test_func).parameters.items():
        if name in INPUT_TYPE_MAP:
            inputs[name] = {"type": INPUT_TYPE_MAP[name]}
        elif name == "args" or name == "kwargs":
            continue
        else:
            params[name] = {
                "type": (
                    arg.annotation.__name__
                    if arg.annotation and hasattr(arg.annotation, "__name__")
                    else None
                ),
                "default": (
                    arg.default if arg.default is not inspect.Parameter.empty else None
                ),
            }

    return inputs, params


def load_test(
    test_id: str, test_func: Optional[Callable[..., Any]] = None, reload: bool = False
) -> Callable[..., Any]:
    """Load a test by test ID

    Test IDs are in the format `namespace.path_to_module.TestClassOrFuncName[:tag]`.
    The tag is optional and is used to distinguish between multiple results from the
    same test.

    Args:
        test_id (str): The test ID in the format `namespace.path_to_module.TestName[:tag]`
        test_func (callable, optional): The test function to load. If not provided, the
            test will be loaded from the test provider. Defaults to None.
        reload (bool, optional): If True, reload the test even if it's already loaded.
            Defaults to False.
    """
    # remove tag if present
    test_id = test_id.split(":", 1)[0]
    namespace = test_id.split(".", 1)[0]

    # if not already loaded, load it from appropriate provider
    if test_id not in test_store.tests or reload:
        if test_id.startswith("validmind.composite_metric"):
            # TODO: add composite metric loading
            pass

        if not test_func:
            if not test_provider_store.has_test_provider(namespace):
                raise LoadTestError(
                    f"No test provider found for namespace: {namespace}"
                )

            provider = test_provider_store.get_test_provider(namespace)

            try:
                test_func = provider.load_test(test_id.split(".", 1)[1])
            except Exception as e:
                raise LoadTestError(
                    f"Unable to load test '{test_id}' from {namespace} test provider",
                    original_error=e,
                ) from e

        # add test_id as an attribute to the test function
        test_func.test_id = test_id

        # fallback to using func name if no docstring is found
        if not inspect.getdoc(test_func):
            test_func.__doc__ = f"{test_func.__name__} ({test_id})"

        # add inputs and params as attributes to the test function
        test_func.inputs, test_func.params = _inspect_signature(test_func)

        # ensure tags and tasks attributes exist, default to empty list if not present
        if not hasattr(test_func, "__tags__"):
            test_func.__tags__ = []
        if not hasattr(test_func, "__tasks__"):
            test_func.__tasks__ = []

        test_store.register_test(test_id, test_func)

    return test_store.get_test(test_id)


def _list_test_ids() -> List[str]:
    """List all available test IDs"""
    test_ids = []

    for namespace, test_provider in test_provider_store.test_providers.items():
        test_ids.extend(
            [f"{namespace}.{test_id}" for test_id in sorted(test_provider.list_tests())]
        )

    return test_ids


def _load_tests(test_ids: List[str]) -> Dict[str, Callable[..., Any]]:
    """Load a set of tests, handling missing dependencies."""
    tests = {}

    for test_id in test_ids:
        try:
            tests[test_id] = load_test(test_id)
        except LoadTestError as e:
            if not e.original_error or not isinstance(
                e.original_error, MissingDependencyError
            ):
                raise e

            e = e.original_error

            logger.debug(str(e))

            if e.extra:
                logger.debug(
                    f"Skipping `{test_id}` as it requires extra dependencies: {e.required_dependencies}."
                    f" Please run `pip install validmind[{e.extra}]` to view and run this test."
                )
            else:
                logger.debug(
                    f"Skipping `{test_id}` as it requires missing dependencies: {e.required_dependencies}."
                    " Please install the missing dependencies to view and run this test."
                )

    return tests


def _test_description(test_description: str, num_lines: int = 5) -> str:
    """Format a test description"""
    description = test_description.strip("\n").strip()

    if len(description.split("\n")) > num_lines:
        return description.strip().split("\n")[0] + "..."

    return description


def _pretty_list_tests(
    tests: Dict[str, Callable[..., Any]], truncate: bool = True
) -> None:
    """Pretty print a list of tests"""
    rows = []
    for test_id, test in tests.items():
        has_figure, has_table = _inspect_return_type(
            inspect.signature(test).return_annotation
        )
        rows.append(
            {
                "ID": test_id,
                "Name": test_id_to_name(test_id),
                "Description": _test_description(
                    inspect.getdoc(test),
                    num_lines=(5 if truncate else 999999),
                ),
                "Has Figure": has_figure,
                "Has Table": has_table,
                "Required Inputs": list(test.inputs.keys()),
                "Params": test.params,
                "Tags": test.__tags__,
                "Tasks": test.__tasks__,
            }
        )

    return format_dataframe(pd.DataFrame(rows))


def list_tags() -> List[str]:
    """List all unique available tags"""

    unique_tags = set()

    for test in _load_tests(list_tests(pretty=False)).values():
        unique_tags.update(test.__tags__)

    return list(unique_tags)


def list_tasks_and_tags(as_json: bool = False) -> Union[str, Dict[str, List[str]]]:
    """
    List all task types and their associated tags, with one row per task type and
    all tags for a task type in one row.

    Returns:
        pandas.DataFrame: A DataFrame with 'Task Type' and concatenated 'Tags'.
    """
    task_tags_dict = {}

    for test in _load_tests(list_tests(pretty=False)).values():
        for task in test.__tasks__:
            task_tags_dict.setdefault(task, set()).update(test.__tags__)

    if as_json:
        return task_tags_dict

    return format_dataframe(
        pd.DataFrame(
            [
                {"Task": task, "Tags": ", ".join(tags)}
                for task, tags in task_tags_dict.items()
            ]
        )
    )


def list_tasks() -> List[str]:
    """List all unique available tasks"""
    unique_tasks = set()

    for test in _load_tests(list_tests(pretty=False)).values():
        unique_tasks.update(test.__tasks__)

    return list(unique_tasks)


def list_tests(
    filter: Optional[str] = None,
    task: Optional[str] = None,
    tags: Optional[List[str]] = None,
    pretty: bool = True,
    truncate: bool = True,
) -> Union[List[str], None]:
    """List all tests in the tests directory.

    Args:
        filter (str, optional): Find tests where the ID, tasks or tags match the
            filter string. Defaults to None.
        task (str, optional): Find tests that match the task. Can be used to
            narrow down matches from the filter string. Defaults to None.
        tags (list, optional): Find tests that match list of tags. Can be used to
            narrow down matches from the filter string. Defaults to None.
        pretty (bool, optional): If True, returns a pandas DataFrame with a
            formatted table. Defaults to True.
        truncate (bool, optional): If True, truncates the test description to the first
            line. Defaults to True. (only used if pretty=True)
    """
    test_ids = _list_test_ids()

    # no need to load test funcs (takes a while) if we're just returning the test ids
    if not filter and not task and not tags and not pretty:
        return test_ids

    tests = _load_tests(test_ids)

    # first search by the filter string since it's the most general search
    if filter is not None:
        tests = {
            test_id: test
            for test_id, test in tests.items()
            if filter.lower() in test_id.lower()
            or any(filter.lower() in task.lower() for task in test.__tasks__)
            or any(fuzzy_match(tag, filter.lower()) for tag in test.__tags__)
        }

    # then filter by task type and tags since they are more specific
    if task is not None:
        tests = {
            test_id: test for test_id, test in tests.items() if task in test.__tasks__
        }

    if tags is not None:
        tests = {
            test_id: test
            for test_id, test in tests.items()
            if all(tag in test.__tags__ for tag in tags)
        }

    if not pretty:
        return list(tests.keys())

    return _pretty_list_tests(tests, truncate=truncate)


def describe_test(
    test_id: Optional[TestID] = None, raw: bool = False, show: bool = True
) -> Union[str, Dict[str, Any]]:
    """Get or show details about the test

    This function can be used to see test details including the test name, description,
    required inputs and default params. It can also be used to get a dictionary of the
    above information for programmatic use.

    Args:
        test_id (str, optional): The test ID. Defaults to None.
        raw (bool, optional): If True, returns a dictionary with the test details.
            Defaults to False.
    """
    test = load_test(test_id)

    details = {
        "ID": test_id,
        "Name": test_id_to_name(test_id),
        "Required Inputs": test.inputs or [],
        "Params": test.params or {},
        "Description": inspect.getdoc(test).strip() or "",
    }

    if raw:
        return details

    html = test_content_block_html.format(
        test_id=test_id,
        uuid=str(uuid4()),
        title=f'{details["Name"]}',
        description=md_to_html(details["Description"].strip()),
        required_inputs=", ".join(details["Required Inputs"] or ["None"]),
        params_table="\n".join(
            [
                f"<tr><td>{param}</td><td>{pformat(param_spec['default'], indent=4)}</td></tr>"
                for param, param_spec in details["Params"].items()
            ]
        ),
        table_display="table" if details["Params"] else "none",
        example_inputs=json.dumps(
            {name: f"my_vm_{name}" for name in (details["Required Inputs"] or [])},
            indent=4,
        ),
        example_params=json.dumps(
            {param: f"my_vm_{param}" for param in (details["Params"] or {}).keys()},
            indent=4,
        ),
        instructions_display="block" if show else "none",
    )

    if not show:
        return html

    from ..vm_models.html_renderer import StatefulHTMLRenderer

    accordion_html = StatefulHTMLRenderer.render_accordion(
        items=[html],
        titles=[f"Test: {details['Name']} ('{test_id}')"],
    )
    display(accordion_html)
