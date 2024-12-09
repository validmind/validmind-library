# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Result Objects for test results
"""
import asyncio
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import pandas as pd
from ipywidgets import HTML, VBox

from ... import api_client
from ...ai.utils import DescriptionFuture
from ...logging import get_logger
from ...utils import NumpyEncoder, display, run_async, test_id_to_name
from ..figure import Figure
from ..input import VMInput
from .utils import (
    AI_REVISION_NAME,
    DEFAULT_REVISION_NAME,
    check_for_sensitive_data,
    figures_to_widgets,
    get_result_template,
    tables_to_widgets,
    update_metadata,
)

logger = get_logger(__name__)


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
    name: str = None

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

    name: str = "Failed Test"
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

    name: str = "Test Result"
    ref_id: str = None
    title: Optional[str] = None
    description: Optional[Union[str, DescriptionFuture]] = None
    metric: Optional[Union[int, float]] = None
    tables: Optional[List[ResultTable]] = None
    figures: Optional[List[Figure]] = None
    passed: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Union[List[VMInput], VMInput]]] = None
    metadata: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    _was_description_generated: bool = False
    _unsafe: bool = False

    @property
    def test_name(self) -> str:
        """Get the test name, using custom title if available."""
        return self.title or test_id_to_name(self.result_id)

    def __repr__(self) -> str:
        attrs = [
            attr
            for attr in [
                "description",
                "params",
                "tables",
                "figures",
                "metric",
                "passed",
            ]
            if getattr(self, attr) is not None
        ]

        return f'TestResult("{self.result_id}", {", ".join(attrs)})'

    def __post_init__(self):
        if self.ref_id is None:
            self.ref_id = str(uuid4())

    def _get_flat_inputs(self):
        # remove duplicates by `input_id`
        inputs = {}

        for input_or_list in self.inputs.values():
            if isinstance(input_or_list, list):
                inputs.update({input.input_id: input for input in input_or_list})
            else:
                inputs[input_or_list.input_id] = input_or_list

        return list(inputs.values())

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
            return HTML(f"<h3>{self.test_name}: <code>{self.metric}</code></h3>")

        template_data = {
            "test_name": self.test_name,
            "passed_icon": "" if self.passed is None else "✅" if self.passed else "❌",
            "description": self.description.replace("h3", "strong"),
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
        rendered = get_result_template().render(**template_data)

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
            "title": self.title,
            "ref_id": self.ref_id,
            "params": self.params,
            "inputs": [_input.input_id for _input in self._get_flat_inputs()],
            "passed": self.passed,
            "summary": [table.serialize() for table in (self.tables or [])],
            "metadata": self.metadata,
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
