# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Result Wrappers for test and metric results
"""
import asyncio
import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import pandas as pd
from ipywidgets import HTML, GridBox, Layout, VBox
from jinja2 import Template

from .. import api_client
from ..ai.utils import DescriptionFuture
from ..logging import get_logger
from ..utils import NumpyEncoder, display, run_async, test_id_to_name
from .dataset import VMDataset
from .figure import Figure
from .input import VMInput

logger = get_logger(__name__)


AI_REVISION_NAME = "Generated by ValidMind AI"
DEFAULT_REVISION_NAME = "Default Description"

_result_template = None


def _get_result_template():
    global _result_template

    if _result_template is None:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "html_templates",
            "result.jinja",
        )

        with open(file_path) as f:
            _result_template = Template(f.read())

    return _result_template


def check_for_sensitive_data(data: pd.DataFrame, inputs: List[VMInput]):
    """Check if a table contains raw data from input datasets"""
    dataset_columns = {
        col: len(input_obj.df)
        for input_obj in inputs
        if isinstance(input_obj, VMDataset)
        for col in input_obj.columns
    }

    table_columns = {col: len(data) for col in data.columns}

    offending_columns = [
        col
        for col in table_columns
        if col in dataset_columns and table_columns[col] == dataset_columns[col]
    ]

    if offending_columns:
        raise ValueError(
            f"Raw input data found in table, pass `unsafe=True` "
            f"or remove the offending columns: {offending_columns}"
        )


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
            md = await api_client.aget_metadata(content_id)
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

        await api_client.alog_metadata(content_id, text, _json)


def tables_to_widgets(tables: List["ResultTable"]):
    """Convert summary (list of json tables) into a list of ipywidgets"""
    widgets = [
        HTML("<h3>Tables</h3>"),
    ]

    for table in tables:
        html = ""
        if table.title:
            html += f"<h4>{table.title}</h4>"

        html += (
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

        widgets.append(HTML(html))

    return widgets


def figures_to_widgets(figures: List[Figure]) -> list:
    """Plot figures to a ipywidgets GridBox"""
    num_columns = 2 if len(figures) > 1 else 1

    plot_widgets = GridBox(
        [figure.to_widget() for figure in figures],
        layout=Layout(
            grid_template_columns=f"repeat({num_columns}, 1fr)",
        ),
    )

    return [HTML("<h3>Figures</h3>"), plot_widgets]


@dataclass
class ResultTable:
    """
    A dataclass that holds the table summary of result
    """

    data: Union[List[Any], pd.DataFrame]
    title: str

    def __repr__(self) -> str:
        return f'ResultTable(title="{self.title}")' if self.title else "ResultTable"

    def __post_init__(self):
        if isinstance(self.data, list):
            self.data = pd.DataFrame(self.data)

        self.data = self.data.round(4)

    def serialize(self):
        data = {
            "type": "table",
            "data": self.data.to_dict(orient="records"),
        }

        if self.title:
            data["metadata"] = {"title": self.title}

        return data


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

    ref_id: str = None
    description: Optional[Union[str, DescriptionFuture]] = None
    metric: Optional[Union[int, float]] = None
    tables: Optional[List[ResultTable]] = None
    figures: Optional[List[Figure]] = None
    passed: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Union[List[VMInput], VMInput]]] = None

    _was_description_generated: bool = False
    _unsafe: bool = False

    def __repr__(self) -> str:
        artifacts = []
        if self.description:
            artifacts.append("description")
        if self.params:
            artifacts.append("params")
        if self.tables:
            artifacts.append("tables")
        if self.figures:
            artifacts.append("figures")
        if self.metric:
            artifacts.append("metric")
        if self.passed:
            artifacts.append("passed")

        return f'TestResult("{self.result_id}", {", ".join(artifacts)})'

    def __post_init__(self):
        if self.ref_id is None:
            self.ref_id = str(uuid4())

    def _get_flat_inputs(self):
        return {
            _input
            for _item in self.inputs.values()
            for _input in (_item if isinstance(_item, list) else [_item])
        }

    def add_table(self, table: ResultTable):
        if self.tables is None:
            self.tables = []
        self.tables.append(table)

    def add_figure(self, figure: Figure):
        if self.figures is None:
            self.figures = []

        if figure.ref_id != self.ref_id:
            figure.ref_id = self.ref_id

        self.figures.append(figure)

    def to_widget(self):
        if isinstance(self.description, DescriptionFuture):
            self.description = self.description.get_description()
            self._was_description_generated = True

        if self.metric is not None and not self.tables and not self.figures:
            return HTML(f"<h3>{self.result_id}: <code>{self.metric}</code></h3>")

        template_data = {
            "test_name": test_id_to_name(self.result_id),
            "passed_icon": "" if self.passed is None else "✅" if self.passed else "❌",
            "description": self.description.replace("h3", "strong"),
            # TODO: add inputs
            "params": (
                json.dumps(self.params, cls=NumpyEncoder, indent=2)
                if self.params
                else None
            ),
            "show_metric": self.metric is not None,
            "metric": self.metric,
            "tables": self.tables,
            "figures": self.figures,
        }
        rendered = _get_result_template().render(**template_data)

        widgets = [HTML(rendered)]

        if self.tables:
            widgets.extend(tables_to_widgets(self.tables))
        if self.figures:
            widgets.extend(figures_to_widgets(self.figures))

        return VBox(widgets)

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

    def serialize(self):
        """Serialize the result for the API"""
        return {
            "test_name": self.result_id,
            "ref_id": self.ref_id,
            "params": self.params,
            "inputs": [_input.input_id for _input in self._get_flat_inputs()],
            "passed": self.passed,
            "summary": [table.serialize() for table in (self.tables or [])],
        }

    async def log_async(
        self, section_id: str = None, position: int = None, unsafe: bool = False
    ):
        tasks = []  # collect tasks to run in parallel (async)

        if self.metric is not None:
            # metrics are logged as separate entities
            tasks.append(
                api_client.alog_metric(
                    key=self.result_id,
                    value=self.metric,
                    inputs=[input.input_id for input in self._get_flat_inputs()],
                    params=self.params,
                )
            )

        if self.tables or self.figures:
            tasks.append(
                api_client.alog_test_result(
                    result=self.serialize(),
                    section_id=section_id,
                    position=position,
                )
            )

            tasks.extend(
                [api_client.alog_figure(figure) for figure in (self.figures or [])]
            )

            if self.description:
                if isinstance(self.description, DescriptionFuture):
                    self.description = self.description.get_description()
                    self._was_description_generated = True

                revision_name = (
                    AI_REVISION_NAME
                    if self._was_description_generated
                    else DEFAULT_REVISION_NAME
                )

                tasks.append(
                    update_metadata(
                        content_id=f"test_description:{self.result_id}::{revision_name}",
                        text=self.description,
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
        if not unsafe:
            for table in self.tables or []:
                check_for_sensitive_data(table.data, self._get_flat_inputs())

        if section_id:
            self._validate_section_id_for_block(section_id, position)

        run_async(self.log_async, section_id=section_id, position=position)