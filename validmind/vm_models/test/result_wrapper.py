# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Result Wrappers for test and metric results
"""
import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd
from ipywidgets import HTML, GridBox, Layout, VBox

from ... import api_client
from ...ai.test_descriptions import AI_REVISION_NAME, DescriptionFuture
from ...input_registry import input_registry
from ...logging import get_logger
from ...utils import NumpyEncoder, display, run_async, test_id_to_name
from ..dataset import VMDataset
from ..figure import Figure
from .metric_result import MetricResult
from .output_template import OutputTemplate
from .result_summary import ResultSummary
from .threshold_test_result import ThresholdTestResults

logger = get_logger(__name__)


async def update_metadata(content_id: str, text: str, _json: Union[Dict, List] = None):
    """Create or Update a Metadata Object"""
    parts = content_id.split("::")
    content_id = parts[0]
    revision_name = parts[1] if len(parts) > 1 else None

    # we always want composite metric definitions to be updated
    should_update = content_id.startswith("composite_metric_def:")

    # if we are updating a metric or test description, we check if the text
    # has changed from the last time it was logged, and only update if it has
    if content_id.split(":", 1)[0] in ["metric_description", "test_description"]:
        try:
            md = await api_client.get_metadata(content_id)
            # if there is an existing description, only update it if the new one
            # is different and is an AI-generated description
            should_update = (
                md["text"] != text if revision_name == AI_REVISION_NAME else False
            )
            logger.debug(f"Check if description has changed: {should_update}")
        except Exception:
            # if exception, assume its not created yet TODO: don't catch all
            should_update = True

    if should_update:
        if revision_name:
            content_id = f"{content_id}::{revision_name}"

        logger.debug(f"Updating metadata for `{content_id}`")

        await api_client.log_metadata(content_id, text, _json)


def plot_figures(figures: List[Figure]) -> None:
    """Plot figures to a ipywidgets GridBox"""
    plots = [figure.to_widget() for figure in figures]
    num_columns = 2 if len(figures) > 1 else 1

    return GridBox(
        plots,
        layout=Layout(grid_template_columns=f"repeat({num_columns}, 1fr)"),
    )


def _summary_tables_to_widget(summary: ResultSummary):
    """Convert summary (list of json tables) into ipywidgets"""
    widgets = []

    for table in summary.results:
        if table.metadata and table.metadata.title:
            widgets.append(HTML(f"<h4>{table.metadata.title}</h4>"))

        df_html = (
            pd.DataFrame(table.data)
            .style.format(precision=4)
            .hide(axis="index")
            .set_table_styles(
                [
                    {
                        "selector": "",
                        "props": [("width", "100%")],
                    },
                    {
                        "selector": "th",
                        "props": [("text-align", "left")],
                    },
                    {
                        "selector": "tbody tr:nth-child(even)",
                        "props": [("background-color", "#FFFFFF")],
                    },
                    {
                        "selector": "tbody tr:nth-child(odd)",
                        "props": [("background-color", "#F5F5F5")],
                    },
                    {
                        "selector": "td, th",
                        "props": [
                            ("padding-left", "5px"),
                            ("padding-right", "5px"),
                        ],
                    },
                ]
            )
            .set_properties(**{"text-align": "left"})
            .to_html(escape=False)
        )
        widgets.append(HTML(df_html))

    return widgets


@dataclass
class ResultWrapper(ABC):
    """Base Class for test suite results"""

    name: str = "ResultWrapper"
    # id of the result, can be set by the subclass. This helps
    # looking up results later on
    result_id: str = None
    # Text description from test or metric (docstring usually)
    result_description: str = None
    # Text metadata about the result, can include description, etc.
    result_metadata: List[dict] = None
    # Output template to use for rendering the result
    output_template: Optional[str] = None

    def __str__(self) -> str:
        """May be overridden by subclasses"""
        return self.__class__.__name__

    @abstractmethod
    def to_widget(self):
        """Create an ipywdiget representation of the result... Must be overridden by subclasses"""
        raise NotImplementedError

    def render(self, output_template=None):
        """Helper method thats lets the user try out output templates"""
        if output_template:
            self.output_template = output_template

        return self.to_widget()

    def _validate_section_id_for_block(self, section_id: str, position: int = None):
        """
        Validate the section_id exits on the template before logging. We validate
        if the section exists and if the user provided position is within the bounds
        of the section. When the position is None, we assume it goes to the end of the section.
        """
        if section_id is None:
            return

        api_client.reload()
        found = False
        client_config = api_client.client_config

        for section in client_config.documentation_template["sections"]:
            if section["id"] == section_id:
                found = True
                break

        if not found:
            raise ValueError(
                f"Section with id {section_id} not found in the model's document"
            )

        # Check if the block already exists in the section
        block_definition = {
            "content_id": self.result_id,
            "content_type": (
                "metric" if isinstance(self, MetricResultWrapper) else "test"
            ),
        }
        blocks = section.get("contents", [])
        for block in blocks:
            if (
                block["content_id"] == block_definition["content_id"]
                and block["content_type"] == block_definition["content_type"]
            ):
                logger.info(
                    f"Test driven block with content_id {block_definition['content_id']} already exists in the document's section"
                )
                return

        # Validate that the position is within the bounds of the section
        if position is not None:
            num_blocks = len(blocks)
            if position < 0 or position > num_blocks:
                raise ValueError(
                    f"Invalid position {position}. Must be between 0 and {num_blocks}"
                )

    def show(self):
        """Display the result... May be overridden by subclasses"""
        display(self.to_widget())

    @abstractmethod
    async def log_async(self):
        """Log the result... Must be overridden by subclasses"""
        raise NotImplementedError

    def log(self, section_id: str = None, position: int = None):
        """Log the result... May be overridden by subclasses"""

        self._validate_section_id_for_block(section_id, position)
        run_async(self.log_async, section_id=section_id, position=position)


@dataclass
class FailedResultWrapper(ResultWrapper):
    """
    Result wrapper for test suites that fail to load or run properly
    """

    name: str = "Failed"
    error: Exception = None
    message: str = None

    def __repr__(self) -> str:
        return f'FailedResult(result_id="{self.result_id}")'

    def to_widget(self):
        return HTML(f"<h3 style='color: red;'>{self.message}</h3><p>{self.error}</p>")

    async def log_async(self):
        pass


@dataclass
class MetricResultWrapper(ResultWrapper):
    """
    Result wrapper for metrics that run as part of a test suite
    """

    name: str = "Metric"
    scalar: Optional[Union[int, float]] = None
    metric: Optional[MetricResult] = None
    figures: Optional[List[Figure]] = None
    inputs: List[str] = None  # List of input ids
    params: Dict = None

    def __repr__(self) -> str:
        if self.metric:
            return f'{self.__class__.__name__}(result_id="{self.result_id}", metric, figures)'
        else:
            return f'{self.__class__.__name__}(result_id="{self.result_id}", figures)'

    def to_widget(self):
        if self.metric and self.metric.key == "dataset_description":
            return ""

        vbox_children = [
            HTML(f"<h1>{test_id_to_name(self.result_id)}</h1>"),
        ]

        if self.result_metadata:
            metric_description = self.result_metadata[0].get("text", "")
            if isinstance(metric_description, DescriptionFuture):
                metric_description = metric_description.get_description()
                self.result_metadata[0]["text"] = metric_description

            vbox_children.append(HTML(metric_description))

        if self.scalar is not None:
            vbox_children.append(
                HTML(
                    "<h3>Unit Metrics</h3>"
                    f"<p>{test_id_to_name(self.result_id)} "
                    f"(<i>{self.result_id}</i>): "
                    f"<code>{self.scalar}</code></p>"
                )
            )

        if self.metric:
            vbox_children.append(HTML("<h3>Tables</h3>"))
            if self.output_template:
                vbox_children.append(
                    HTML(
                        OutputTemplate(self.output_template).render(
                            value=self.metric.value
                        )
                    )
                )
            elif self.metric.summary:
                vbox_children.extend(_summary_tables_to_widget(self.metric.summary))

        if self.figures:
            vbox_children.append(HTML("<h3>Plots</h3>"))
            plot_widgets = plot_figures(self.figures)
            vbox_children.append(plot_widgets)

        return VBox(vbox_children)

    def _get_filtered_summary(self):
        """Check if the metric summary has columns from input datasets with matching row counts."""
        dataset_columns = self._get_dataset_columns()
        filtered_results = []

        for table in self.metric.summary.results:
            table_columns = self._get_table_columns(table)
            sensitive_columns = self._find_sensitive_columns(
                dataset_columns, table_columns
            )

            if sensitive_columns:
                self._log_sensitive_data_warning(sensitive_columns)
            else:
                filtered_results.append(table)

        self.metric.summary.results = filtered_results
        return self.metric.summary

    def _get_dataset_columns(self):
        dataset_columns = {}
        for input_item in self.inputs:
            input_id = (
                input_item if isinstance(input_item, str) else input_item.input_id
            )
            input_obj = input_registry.get(input_id)
            if isinstance(input_obj, VMDataset):
                dataset_columns.update(
                    {col: len(input_obj.df) for col in input_obj.columns}
                )
        return dataset_columns

    def _get_table_columns(self, table):
        if isinstance(table.data, pd.DataFrame):
            return {col: len(table.data) for col in table.data.columns}
        elif isinstance(table.data, list) and table.data:
            return {col: len(table.data) for col in table.data[0].keys()}
        else:
            raise ValueError("Invalid data type in summary table")

    def _find_sensitive_columns(self, dataset_columns, table_columns):
        return [
            col
            for col, row_count in table_columns.items()
            if col in dataset_columns and row_count == dataset_columns[col]
        ]

    def _log_sensitive_data_warning(self, sensitive_columns):
        logger.warning(
            "Sensitive data in metric summary table. Not logging to API automatically. "
            "Pass `unsafe=True` to result.log() method to override manually."
        )
        logger.warning(
            f"The following columns are present in the table with matching row counts: {sensitive_columns}"
        )

    async def log_async(
        self, section_id: str = None, position: int = None, unsafe=False
    ):
        tasks = []  # collect tasks to run in parallel (async)

        if self.scalar is not None:
            # scalars (unit metrics) are logged as key-value pairs associated with the inventory model
            tasks.append(
                api_client.alog_metric(
                    key=self.result_id,
                    value=self.scalar,
                    inputs=self.inputs,
                    params=self.params,
                )
            )

        if self.metric:
            if self.metric.summary and not unsafe:
                self.metric.summary = self._get_filtered_summary()

            tasks.append(
                api_client.log_metric_result(
                    metric=self.metric,
                    inputs=self.inputs,
                    output_template=self.output_template,
                    section_id=section_id,
                    position=position,
                )
            )

        if self.figures:
            tasks.extend([api_client.log_figure(figure) for figure in self.figures])

        if hasattr(self, "result_metadata") and self.result_metadata:
            description = self.result_metadata[0].get("text", "")
            if isinstance(description, DescriptionFuture):
                description = description.get_description()
                self.result_metadata[0]["text"] = description

            for metadata in self.result_metadata:
                tasks.append(
                    update_metadata(
                        content_id=metadata["content_id"],
                        text=metadata.get("text", ""),
                        _json=metadata.get("json"),
                    )
                )

        return await asyncio.gather(*tasks)


@dataclass
class ThresholdTestResultWrapper(ResultWrapper):
    """
    Result wrapper for test results produced by the tests that run as part of a test suite
    """

    name: str = "Threshold Test"
    figures: Optional[List[Figure]] = None
    test_results: ThresholdTestResults = None
    inputs: List[str] = None

    def __repr__(self) -> str:
        if self.test_results:
            return (
                f'{self.__class__.__name__}(result_id="{self.result_id}", test_results)'
            )
        else:
            return f'{self.__class__.__name__}(result_id="{self.result_id}", figures)'

    def to_widget(self):
        vbox_children = []
        description_html = []

        description_html.append(
            f"""
            <h1>{test_id_to_name(self.test_results.test_name)} {"✅" if self.test_results.passed else "❌"}</h1>
            """
        )

        if self.result_metadata:
            metric_description = self.result_metadata[0].get("text", "")
            if isinstance(metric_description, DescriptionFuture):
                metric_description = metric_description.get_description()
                self.result_metadata[0]["text"] = metric_description

            description_html.append(metric_description)

        test_params = json.dumps(self.test_results.params, cls=NumpyEncoder, indent=2)
        description_html.append(
            f"""
                <h4>Test Parameters</h4>
                <pre>{test_params}</pre>
            """
        )

        vbox_children.append(HTML("".join(description_html)))

        if self.test_results.summary:
            vbox_children.append(HTML("<h3>Tables</h3>"))
            vbox_children.extend(_summary_tables_to_widget(self.test_results.summary))

        if self.figures:
            vbox_children.append(HTML("<h3>Plots</h3>"))
            plot_widgets = plot_figures(self.figures)
            vbox_children.append(plot_widgets)

        return VBox(vbox_children)

    async def log_async(self, section_id: str = None, position: int = None):
        tasks = [
            api_client.log_test_result(
                self.test_results, self.inputs, section_id, position
            )
        ]

        if self.figures:
            tasks.extend([api_client.log_figure(figure) for figure in self.figures])

        if hasattr(self, "result_metadata") and self.result_metadata:
            description = self.result_metadata[0].get("text", "")
            if isinstance(description, DescriptionFuture):
                description = description.get_description()
                self.result_metadata[0]["text"] = description

            for metadata in self.result_metadata:
                tasks.append(update_metadata(metadata["content_id"], metadata["text"]))

        await asyncio.gather(*tasks)
