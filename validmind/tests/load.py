# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Module for listing and loading tests."""

import inspect
import json
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import pandas as pd
from ipywidgets import HTML, Accordion

from ..errors import LoadTestError, MissingDependencyError
from ..html_templates.content_blocks import test_content_block_html
from ..logging import get_logger
from ..utils import display, format_dataframe, fuzzy_match, md_to_html, test_id_to_name
from ..vm_models import VMDataset, VMModel
from .__types__ import TestID
from ._store import test_provider_store, test_store

logger = get_logger(__name__)


INPUT_TYPE_MAP = {
    "dataset": VMDataset,
    "datasets": List[VMDataset],
    "model": VMModel,
    "models": List[VMModel],
}


def _inspect_signature(test_func: Callable[..., Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
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
    test_id: str,
    test_func: Optional[Callable[..., Any]] = None,
    reload: bool = False
) -> Callable[..., Any]:
    """Load a test by test ID

    Test IDs are in the format `namespace.path_to_module.TestClassOrFuncName[:tag]`.
    The tag is optional and is used to distinguish between multiple results from the
    same test.

    Args:
        test_id (str): The test ID in the format `namespace.path_to_module.TestName[:tag]`
        test_func (callable, optional): The test function to load. If not provided, the
            test will be loaded from the test provider. Defaults to None.
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
        except MissingDependencyError as e:
            logger.debug(f"Skipping test {test_id} due to missing dependency: {str(e)}")
    return tests


def _test_description(test_description: str, num_lines: int = 5) -> str:
    """Format a test description"""
    if len(test_description.split("\n")) > num_lines:
        return test_description.strip().split("\n")[0] + "..."
    return test_description


def _pretty_list_tests(tests: Dict[str, Callable[..., Any]], truncate: bool = True) -> None:
    """Pretty print a list of tests"""
    for test_id, test_func in sorted(tests.items()):
        print(f"\n{test_id_to_name(test_id)}")
        if test_func.__doc__:
            print(_test_description(test_func.__doc__, 5 if truncate else None))


def list_tags() -> Set[str]:
    """List all available tags"""
    tags = set()
    for test_func in test_store.tests.values():
        if hasattr(test_func, "__tags__"):
            tags.update(test_func.__tags__)
    return tags


def list_tasks_and_tags(as_json: bool = False) -> Union[str, Dict[str, List[str]]]:
    """List all available tasks and tags"""
    tasks = list(list_tasks())
    tags = list(list_tags())

    if as_json:
        return json.dumps({"tasks": tasks, "tags": tags}, indent=2)

    return {
        "tasks": tasks,
        "tags": tags,
    }


def list_tasks() -> Set[str]:
    """List all available tasks"""
    tasks = set()
    for test_func in test_store.tests.values():
        if hasattr(test_func, "__tasks__"):
            tasks.update(test_func.__tasks__)
    return tasks


def list_tests(
    filter: Optional[str] = None,
    task: Optional[str] = None,
    tags: Optional[List[str]] = None,
    pretty: bool = True,
    truncate: bool = True
) -> Union[Dict[str, Callable[..., Any]], None]:
    """List all available tests with optional filtering"""
    test_ids = _list_test_ids()

    if filter:
        test_ids = [
            test_id
            for test_id in test_ids
            if fuzzy_match(filter, test_id_to_name(test_id))
        ]

    tests = _load_tests(test_ids)

    if task:
        tests = {
            test_id: test_func
            for test_id, test_func in tests.items()
            if hasattr(test_func, "__tasks__") and task in test_func.__tasks__
        }

    if tags:
        tests = {
            test_id: test_func
            for test_id, test_func in tests.items()
            if hasattr(test_func, "__tags__")
            and all(tag in test_func.__tags__ for tag in tags)
        }

    if pretty:
        _pretty_list_tests(tests, truncate=truncate)
        return None

    return tests


def describe_test(
    test_id: Optional[TestID] = None,
    raw: bool = False,
    show: bool = True
) -> Union[str, HTML, Dict[str, Any]]:
    """Describe a test's functionality and parameters"""
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

    display(
        Accordion(
            children=[HTML(html)],
            titles=[f"Test: {details['Name']} ('{test_id}')"],
        )
    )
