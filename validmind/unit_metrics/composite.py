# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import ast
import inspect
from dataclasses import dataclass
from typing import List
from uuid import uuid4

from ..utils import clean_docstring, run_async, test_id_to_name
from ..vm_models.test.metric import Metric
from ..vm_models.test.metric_result import MetricResult
from ..vm_models.test.result_summary import ResultSummary, ResultTable
from ..vm_models.test.result_wrapper import MetricResultWrapper
from . import _get_metric_class, run_metric


def _extract_class_methods(cls):
    source = inspect.getsource(cls)
    tree = ast.parse(source)

    class MethodVisitor(ast.NodeVisitor):
        def __init__(self):
            self.methods = {}

        def visit_FunctionDef(self, node):
            self.methods[node.name] = node
            self.generic_visit(node)

    visitor = MethodVisitor()
    visitor.visit(tree)

    return visitor.methods


def _extract_required_inputs(cls):
    methods = _extract_class_methods(cls)

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.properties = set()
            self.visited_methods = set()

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "inputs":
                self.properties.add(node.attr)

            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id == "self" and node.func.attr in methods:
                    method_name = node.func.attr

                    if method_name not in self.visited_methods:
                        self.visited_methods.add(method_name)
                        self.visit(methods[method_name])

            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(methods["run"])

    return visitor.properties


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
            params=self.params,
            output_template=self.output_template,
            show=False,
        )

        return self.result

    def summary(self, result: dict):
        return ResultSummary(results=[ResultTable(data=[result])])


def load_composite_metric(
    test_id: str = None,
    metric_name: str = None,
    unit_metrics: List[str] = None,
    output_template: str = None,
) -> CompositeMetric:
    # this function can either create a composite metric from a list of unit metrics or
    # load a stored composite metric based on the test id

    # TODO: figure out this circular import thing:
    from ..api_client import get_metadata

    if test_id:
        # get the unit metric ids and output template (if any) from the metadata
        unit_metrics = run_async(
            get_metadata, f"composite_metric_def:{test_id}:unit_metrics"
        )["json"]
        output_template = run_async(
            get_metadata, f"composite_metric_def:{test_id}:output_template"
        )["json"]["output_template"]

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
        metric_cls = _get_metric_class(metric_id)
        # required_inputs.update(_extract_required_inputs(metric_cls))
        required_inputs.update(metric_cls.required_inputs or [])

    class_def.required_inputs = list(required_inputs)

    return class_def


def run_metrics(
    name: str = None,
    metric_ids: List[str] = None,
    description: str = None,
    output_template: str = None,
    inputs: dict = None,
    params: dict = None,
    test_id: str = None,
    show: bool = True,
) -> MetricResultWrapper:
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
        output_template (_type_, optional): Output template to customize the result
            table.
        inputs (_type_, optional): Inputs to pass to the unit metrics. Defaults to None
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
        result = run_metric(
            metric_id=metric_id,
            inputs=inputs,
            params=params,
        )
        results[list(result.summary.keys())[0]] = result.value

    test_id = f"validmind.composite_metric.{name}" if not test_id else test_id

    if not output_template:

        def row(key):
            return f"""
            <tr>
                <td><strong>{key.upper()}</strong></td>
                <td>{{{{ value['{key}'] | number }}}}</td>
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
                {"".join([row(key) for key in results.keys()])}
            </tbody>
        </table>
        """

    result_wrapper = MetricResultWrapper(
        result_id=test_id,
        result_metadata=[
            {
                "content_id": f"metric_description:{test_id}",
                "text": clean_docstring(description),
            },
            {
                "content_id": f"composite_metric_def:{test_id}:unit_metrics",
                "json": metric_ids,
            },
            {
                "content_id": f"composite_metric_def:{test_id}:output_template",
                "json": {"output_template": output_template},
            },
        ],
        inputs=list(inputs.keys()),
        output_template=output_template,
        metric=MetricResult(
            key=test_id,
            ref_id=str(uuid4()),
            value=results,
            summary=ResultSummary(results=[ResultTable(data=[results])]),
        ),
    )

    if show:
        result_wrapper.show()

    return result_wrapper