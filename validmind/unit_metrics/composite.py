# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import List

from ..ai.test_descriptions import get_result_description
from ..logging import get_logger
from ..utils import test_id_to_name
from ..vm_models.result import ResultTable, TestResult
from . import run_metric

logger = get_logger(__name__)


def run_metrics(
    name: str = None,
    metric_ids: List[str] = None,
    description: str = None,
    inputs: dict = None,
    accessed_inputs: List[str] = None,
    params: dict = None,
    test_id: str = None,
    show: bool = True,
    generate_description: bool = True,
) -> TestResult:
    """Run a composite metric

    Composite metrics are metrics that are composed of multiple unit metrics. This
    works by running individual unit metrics and then combining the results into a
    single "MetricResult" object that can be logged and displayed just like any other
    metric result. The special thing about composite metrics is that when they are
    logged to the platform, metadata describing the unit metrics and output template
    used to generate the composite metric is also logged. This means that by grabbing
    the metadata for a composite metric (identified by the test ID
    `validmind.composite_metric.<name>`) the framework can rebuild and rerun it at
    any time.

    Args:
        name (str, optional): Name of the composite metric. Required if test_id is not
            provided. Defaults to None.
        metric_ids (list[str]): List of unit metric IDs to run. Required.
        description (str, optional): Description of the composite metric. Defaults to
            None.
        inputs (_type_, optional): Inputs to pass to the unit metrics. Defaults to None
        accessed_inputs (_type_, optional): Inputs that were accessed when running the
            unit metrics - used for input tracking. Defaults to None.
        params (_type_, optional): Parameters to pass to the unit metrics. Defaults to
            None.
        test_id (str, optional): Test ID of the composite metric. Required if name is
            not provided. Defaults to None.
        show (bool, optional): Whether to show the result immediately. Defaults to True

    Raises:
        ValueError: If metric_ids is not provided
        ValueError: If name or key is not provided

    Returns:
        MetricResultWrapper: The result wrapper object
    """
    if not metric_ids:
        raise ValueError("metric_ids must be provided")

    if not name and not test_id:
        raise ValueError("name or key must be provided")

    # if name is provided, make sure to squash it into a camel case string
    if name:
        name = "".join(word[0].upper() + word[1:] for word in name.split())

    results = {}

    for metric_id in metric_ids:
        metric_name = test_id_to_name(metric_id)
        results[metric_name] = run_metric(
            metric_id=metric_id,
            inputs=inputs,
            params=params,
            show=False,
            value_only=True,
        )

    test_id = f"validmind.composite_metric.{name}" if not test_id else test_id

    tables = [ResultTable(data=[results])]

    result_wrapper = TestResult(
        result_id=test_id,
        result_metadata=[
            get_result_description(
                test_id=test_id,
                test_description=description,
                tables=tables,
                should_generate=generate_description,
            ),
            {
                "content_id": f"composite_metric_def:{test_id}:unit_metrics",
                "json": metric_ids,
            },
        ],
        tables=tables,
        inputs=accessed_inputs,
    )

    if show:
        result_wrapper.show()

    return result_wrapper
