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
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from ipywidgets import HTML, GridBox, Layout, VBox

from .. import api_client
from ..ai.test_descriptions import AI_REVISION_NAME, DescriptionFuture
from ..input_registry import input_registry
from ..logging import get_logger
from ..utils import NumpyEncoder, display, run_async, test_id_to_name
from .dataset import VMDataset
from .figure import Figure
from .output_template import OutputTemplate

logger = get_logger(__name__)


@dataclass
class ResultTable:
    """
    A dataclass that holds the table summary of result
    """

    data: Union[List[Any], pd.DataFrame]
    title: str

    def __post_init__(self):
        if isinstance(self.data, list):
            self.data = pd.DataFrame(self.data)

        self.data = self.data.round(4)

    def serialize(self):
        return {
            "data": self.data.to_dict(orient="records"),
            "metadata": {
                "title": self.title,
            },
        }


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


def _tables_to_widgets(tables: List[ResultTable]):
    """Convert summary (list of json tables) into a list of ipywidgets"""
    widgets = []

    for table in tables:
        if table.title:
            widgets.append(HTML(f"<h4>{table.title}</h4>"))

        df_html = (
            table.data.style.format(precision=4)
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
class Result:
    """Base Class for test suite results"""

    result_id: str = None

    def __str__(self) -> str:
        """May be overridden by subclasses"""
        return self.__class__.__name__

    @abstractmethod
    def to_widget(self):
        """Create an ipywdiget representation of the result... Must be overridden by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def log(self):
        """Log the result... Must be overridden by subclasses"""
        raise NotImplementedError

    def show(self):
        """Display the result... May be overridden by subclasses"""
        display(self.to_widget())


@dataclass
class ErrorResult(Result):
    """Result for test suites that fail to load or run properly"""

    error: Exception = None
    message: str = None

    def __repr__(self) -> str:
        return f'ErrorResult(result_id="{self.result_id}")'

    def to_widget(self):
        return HTML(f"<h3 style='color: red;'>{self.message}</h3><p>{self.error}</p>")

    async def log_async(self):
        pass


@dataclass
class TestResult(Result):
    """Test result"""

    description: Optional[Union[str, DescriptionFuture]] = None
    metric: Optional[Union[int, float]] = None
    tables: Optional[List[ResultTable]] = None
    figures: Optional[List[Figure]] = None
    passed: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    inputs: List[str] = None

    def to_widget(self):  # noqa
        if self.metric and self.metric.key == "dataset_description":
            return ""

        vbox_children = []

        if self.passed is not None:
            title_html = f"""
            <h1>{test_id_to_name(self.result_id)} {"✅" if self.passed else "❌"}</h1>
            """
            vbox_children.append(HTML(title_html))
        else:
            vbox_children.append(HTML(f"<h1>{test_id_to_name(self.result_id)}</h1>"))

        if self.description:
            if isinstance(self.description, DescriptionFuture):
                metric_description = metric_description.get_description()
                self.result_metadata[0]["text"] = metric_description

            vbox_children.append(HTML(metric_description))

        if self.params:
            params_html = f"""
            <h4>Test Parameters</h4>
            <pre>{json.dumps(self.params, cls=NumpyEncoder, indent=2)}</pre>
            """
            vbox_children.append(HTML(params_html))

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
                vbox_children.extend(_tables_to_widgets(self.metric.summary))

        if self.figures:
            vbox_children.append(HTML("<h3>Plots</h3>"))
            plot_widgets = plot_figures(self.figures)
            vbox_children.append(plot_widgets)

        return VBox(vbox_children)

    def _validate_section_id_for_block(
        self, section_id: str, position: Union[int, None] = None
    ):
        """Validate the section_id exits on the template before logging"""
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
            "content_type": "test",
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

    async def log_async(
        self, section_id: str = None, position: int = None, unsafe: bool = False
    ):
        tasks = []  # collect tasks to run in parallel (async)

        if self.metric is not None:
            # scalars (unit metrics) are logged as key-value pairs associated with the inventory model
            tasks.append(
                api_client.alog_metric(
                    key=self.result_id,
                    value=self.metric,
                    inputs=self.inputs,
                    params=self.params,
                )
            )

        if self.tables:
            if not unsafe:
                tables = self._get_filtered_tables()
            else:
                tables = self.tables

            tasks.append(
                api_client.alog_test_result(
                    tables=tables,
                    inputs=self.inputs,
                    section_id=section_id,
                    position=position,
                )
            )

        if self.figures:
            tasks.extend([api_client.log_figure(figure) for figure in self.figures])

        if self.description:
            if isinstance(self.description, DescriptionFuture):
                description = self.description.get_description()
            else:
                description = self.description

            tasks.append(
                update_metadata(
                    content_id=f"test_description:{self.result_id}",
                    text=description,
                )
            )

        return await asyncio.gather(*tasks)

    def log(self, section_id: str = None, position: int = None, unsafe: bool = False):
        """Log the result to ValidMind

        Args:
            section_id (str): The section ID within the model document to insert the
                test result
            position (int): The position (index) within the section to insert the test
                result
            unsafe (bool): If True, log the result even if it contains sensitive data
                i.e. raw data from input datasets
        """
        # do validation before starting as async since that messes with the traceback
        if section_id:
            self._validate_section_id_for_block(section_id, position)

        run_async(self.log_async, section_id=section_id, position=position)
