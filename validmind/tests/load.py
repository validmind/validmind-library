# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Module for listing and loading tests."""

import importlib
import inspect
import json
import sys
from pathlib import Path
from pprint import pformat
from uuid import uuid4

import pandas as pd
from ipywidgets import HTML, Accordion

from ..errors import LoadTestError, MissingDependencyError
from ..html_templates.content_blocks import test_content_block_html
from ..logging import get_logger
from ..unit_metrics.composite import load_composite_metric
from ..utils import (
    NumpyEncoder,
    display,
    format_dataframe,
    fuzzy_match,
    md_to_html,
    test_id_to_name,
)
from .__types__ import TestID
from ._store import test_provider_store, test_store
from .decorator import test as test_decorator
from .utils import test_description

logger = get_logger(__name__)


def __init__():
    directories = [p.name for p in Path(__file__).parent.iterdir() if p.is_dir()]

    for d in directories:
        for path in Path(__file__).parent.joinpath(d).glob("**/**/*.py"):
            if path.name.startswith("__") or not path.name[0].isupper():
                continue  # skip __init__.py and other special files as well as non Test files
            test_id = (
                f"validmind.{d}.{path.parent.stem}.{path.stem}"
                if path.parent.parent.stem == d
                else f"validmind.{d}.{path.stem}"
            )
            test_store.register_test(test_id)


__init__()


def _pretty_list_tests(tests, truncate=True):
    table = [
        {
            "ID": test_id,
            "Name": test_id_to_name(test_id),
            "Description": test_description(test, truncate),
            "Required Inputs": test.required_inputs,
            "Params": test.default_params or {},
        }
        for test_id, test in tests.items()
    ]

    return format_dataframe(pd.DataFrame(table))


def list_tests(
    filter=None, task=None, tags=None, pretty=True, truncate=True, __as_class=False
):
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

    Returns:
        list or pandas.DataFrame: A list of all tests or a formatted table.
    """
    # tests = {
    #     test_id: load_test(test_id, reload=True)
    #     for test_id in test_store.get_test_ids()
    # }
    tests = {}
    for test_id in test_store.get_test_ids():
        try:
            tests[test_id] = load_test(test_id, reload=True)
        except MissingDependencyError as e:
            # skip tests that have missing dependencies
            logger.debug(str(e))

            if e.extra:
                logger.info(
                    f"Skipping `{test_id}` as it requires extra dependencies: {e.required_dependencies}."
                    f" Please run `pip install validmind[{e.extra}]` to view and run this test."
                )
            else:
                logger.info(
                    f"Skipping `{test_id}` as it requires missing dependencies: {e.required_dependencies}."
                    " Please install the missing dependencies to view and run this test."
                )

            continue

    # first search by the filter string since it's the most general search
    if filter is not None:
        tests = {
            test_id: test
            for test_id, test in tests.items()
            if filter.lower() in test_id.lower()
            or any(filter.lower() in task.lower() for task in test.tasks)
            or any(fuzzy_match(tag, filter.lower()) for tag in test.tags)
        }

    # then filter by task type and tags since they are more specific
    if task is not None:
        tests = {test_id: test for test_id, test in tests.items() if task in test.tasks}

    if tags is not None:
        tests = {
            test_id: test
            for test_id, test in tests.items()
            if all(tag in test.tags for tag in tags)
        }

    if __as_class:
        return list(tests.values())

    if not pretty:
        # only return test ids
        return list(tests.keys())

    return _pretty_list_tests(tests, truncate=truncate)


def _load_validmind_test(test_id, reload=False):
    parts = test_id.split(":")[0].split(".")

    test_module = ".".join(parts[1:-1])
    test_class = parts[-1]

    error = None
    test = None

    try:
        full_path = f"validmind.tests.{test_module}.{test_class}"

        if reload and full_path in sys.modules:
            module = importlib.reload(sys.modules[full_path])
        else:
            module = importlib.import_module(full_path)

        test = getattr(module, test_class)
    except ModuleNotFoundError as e:
        error = f"Unable to load test {test_id}. {e}"
    except AttributeError:
        error = f"Unable to load test {test_id}. Test not in module: {test_class}"

    return error, test


def load_test(test_id: str, reload=False):
    """Load a test by test ID

    Test IDs are in the format `namespace.path_to_module.TestClassOrFuncName[:result_id]`.
    The result ID is optional and is used to distinguish between multiple results from the
    running the same test.

    Args:
        test_id (str): The test ID in the format `namespace.path_to_module.TestName[:result_id]`
        reload (bool, optional): Whether to reload the test module. Defaults to False.
    """
    # TODO: we should use a dedicated class for test IDs to handle this consistently
    test_id, result_id = test_id.split(":", 1) if ":" in test_id else (test_id, None)

    error = None
    namespace = test_id.split(".", 1)[0]

    # TODO: lets implement an extensible loading system instead of this ugly if/else
    if test_store.get_custom_test(test_id):
        test = test_store.get_custom_test(test_id)

    elif test_id.startswith("validmind.composite_metric"):
        error, test = load_composite_metric(test_id)

    elif namespace == "validmind":
        error, test = _load_validmind_test(test_id, reload=reload)

    elif test_provider_store.has_test_provider(namespace):
        provider = test_provider_store.get_test_provider(namespace)

        try:
            test = provider.load_test(test_id.split(".", 1)[1])
        except Exception as e:
            error = (
                f"Unable to load test {test_id} from test provider: "
                f"{provider}\n Got Exception: {e}"
            )

    else:
        error = f"Unable to load test {test_id}. No test provider found."

    if error:
        logger.error(error)
        raise LoadTestError(error)

    if inspect.isfunction(test):
        # if its a function, we decorate it and then load the class
        # TODO: simplify this as we move towards all functional metrics
        # "_" is used here so it doesn't conflict with other test ids
        test_decorator("_")(test)
        test = test_store.get_custom_test("_")

    test.test_id = f"{test_id}:{result_id}" if result_id else test_id

    return test


def describe_test(test_id: TestID = None, raw: bool = False, show: bool = True):
    """Get or show details about the test

    This function can be used to see test details including the test name, description,
    required inputs and default params. It can also be used to get a dictionary of the
    above information for programmatic use.

    Args:
        test_id (str, optional): The test ID. Defaults to None.
        raw (bool, optional): If True, returns a dictionary with the test details.
            Defaults to False.
    """
    test = load_test(test_id, reload=True)

    details = {
        "ID": test_id,
        "Name": test_id_to_name(test_id),
        "Required Inputs": test.required_inputs or [],
        "Params": test.default_params or {},
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
                f"<tr><td>{param}</td><td>{pformat(value, indent=4)}</td></tr>"
                for param, value in details["Params"].items()
            ]
        ),
        table_display="table" if details["Params"] else "none",
        example_inputs=json.dumps(
            {name: f"my_vm_{name}" for name in (details["Required Inputs"] or [])},
            indent=4,
        ),
        example_params=json.dumps(details["Params"] or {}, indent=4, cls=NumpyEncoder),
        instructions_display="block" if show else "none",
    )

    if not show:
        return html

    display(
        Accordion(
            children=[HTML(html)],
            titles=[f"Test Description: {details['Name']} ('{test_id}')"],
        )
    )
