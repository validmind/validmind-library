# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass
from typing import List, Tuple, Union
from uuid import uuid4

from ..ai.test_descriptions import get_description_metadata
from ..logging import get_logger
from ..tests.decorator import _inspect_signature
from ..utils import run_async, test_id_to_name
from ..vm_models.test.metric import Metric
from ..vm_models.test.metric_result import MetricResult
from ..vm_models.test.result_summary import ResultSummary, ResultTable
from ..vm_models.test.result_wrapper import MetricResultWrapper
from . import load_metric, run_metric

logger = get_logger(__name__)


@dataclass
class CompositeMetric(Metric):
    unit_metrics: List[str] = None

    def __post_init__(self):
        if self._unit_metrics:
            self.unit_metrics = self._unit_metrics
        elif self.unit_metrics is None:
            raise ValueError("unit_metrics must be provided")

        if hasattr(self, "_output_template") and self._output_template:
            self.output_template = self._output_template

    def run(self):
        self.result = run_metrics(
            test_id=self.test_id,
            metric_ids=self.unit_metrics,
            description=self.description(),
            inputs=self._get_input_dict(),
            accessed_inputs=self.get_accessed_inputs(),
            params=self.params,
            output_template=self.output_template,
            show=False,
            generate_description=self.generate_description,
        )

        return self.result

    def summary(self, result: dict):
        return ResultSummary(results=[ResultTable(data=[result])])


def load_composite_metric(
    test_id: str = None,
    metric_name: str = None,
    unit_metrics: List[str] = None,
    output_template: str = None,
) -> Tuple[Union[None, str], Union[CompositeMetric, None]]:
    # this function can either create a composite metric from a list of unit metrics or
    # load a stored composite metric based on the test id

    # TODO: figure out this circular import thing:
    from ..api_client import get_metadata

    if test_id:
        # get the unit metric ids and output template (if any) from the metadata
        try:
            unit_metrics = run_async(
                get_metadata, f"composite_metric_def:{test_id}:unit_metrics"
            )["json"]
            output_template = run_async(
                get_metadata, f"composite_metric_def:{test_id}:output_template"
            )["json"]["output_template"]
        except Exception:
            return f"Could not load composite metric {test_id}", None

    description = f"""
    Composite metric built from the following unit metrics:
    {', '.join([metric_id.split('.')[-1] for metric_id in unit_metrics])}
    """

    class_def = type(
        test_id.split(".")[-1] if test_id else metric_name,
        (CompositeMetric,),
        {
            "__doc__": description,
            "_unit_metrics": unit_metrics,
            "_output_template": output_template,
        },
    )

    required_inputs = set()
    for metric_id in unit_metrics:
        inputs, _ = _inspect_signature(load_metric(metric_id))
        required_inputs.update(inputs.keys())

    class_def.required_inputs = list(required_inputs)

    return None, class_def


def run_metrics(
    name: str = None,
    metric_ids: List[str] = None,
    description: str = None,
    output_template: str = None,
    inputs: dict = None,
    accessed_inputs: List[str] = None,
    params: dict = None,
    test_id: str = None,
    show: bool = True,
    generate_description: bool = True,
) -> MetricResultWrapper:
    """Run a composite metric

    Composite metrics are metrics that are composed of multiple unit metrics. This
    works by running individual unit metrics and then combining the results into a
    single "MetricResult" object that can be logged and displayed just like any other
    metric result. The special thing about composite metrics is that when they are
    logged to the ValidMind Platform, metadata describing the unit metrics and output template
    used to generate the composite metric is also logged. This means that by grabbing
    the metadata for a composite metric (identified by the test ID
    `validmind.composite_metric.<name>`) the ValidMind Library can rebuild and rerun it at
    any time.

    Args:
        name (str, optional): Name of the composite metric. Required if test_id is not
            provided. Defaults to None.
        metric_ids (list[str]): List of unit metric IDs to run. Required.
        description (str, optional): Description of the composite metric. Defaults to
            None.
        output_template (_type_, optional): Output template to customize the result
            table.
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

    if not output_template:

        def row(name):
            return f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{{{{ value['{name}'] | number }}}}</td>
            </tr>
            """

        output_template = f"""
        <h1{test_id_to_name(test_id)}</h1>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {"".join([row(name) for name in results.keys()])}
            </tbody>
        </table>
        <style>
            th, td {{
                padding: 5px;
                text-align: left;
            }}
        </style>
        """

    result_summary = ResultSummary(results=[ResultTable(data=[results])])
    result_wrapper = MetricResultWrapper(
        result_id=test_id,
        result_metadata=[
            get_description_metadata(
                test_id=test_id,
                default_description=description,
                summary=result_summary.serialize(),
                should_generate=generate_description,
            ),
            {
                "content_id": f"composite_metric_def:{test_id}:unit_metrics",
                "json": metric_ids,
            },
            {
                "content_id": f"composite_metric_def:{test_id}:output_template",
                "json": {"output_template": output_template},
            },
        ],
        inputs=accessed_inputs,
        output_template=output_template,
        metric=MetricResult(
            key=test_id,
            ref_id=str(uuid4()),
            value=results,
            summary=result_summary,
        ),
    )

    if show:
        result_wrapper.show()

    return result_wrapper
