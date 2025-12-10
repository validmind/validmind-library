# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Result objects for test results
"""
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import matplotlib
import pandas as pd
import plotly.graph_objs as go

from ... import api_client
from ...ai.utils import DescriptionFuture
from ...errors import InvalidParameterError
from ...logging import get_logger, log_api_operation
from ...utils import HumanReadableEncoder, display, run_async, test_id_to_name
from ..figure import Figure, create_figure
from ..html_renderer import StatefulHTMLRenderer
from ..input import VMInput
from .pii_filter import PIIDetectionMode, get_pii_detection_mode, scan_df, scan_text
from .utils import (
    AI_REVISION_NAME,
    DEFAULT_REVISION_NAME,
    figures_to_html,
    tables_to_html,
    update_metadata,
)

logger = get_logger(__name__)


class RawData:
    """Holds raw data for a test result."""

    def __init__(self, log: bool = False, **kwargs: Any) -> None:
        """Create a new RawData object.

        Args:
            log (bool): If True, log the raw data to ValidMind.
            **kwargs: Keyword arguments to set as attributes, such as
                `RawData(log=True, dataset_duplicates=df_duplicates)`.
        """
        self.log = log

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"RawData({', '.join(self.__dict__.keys())})"

    def inspect(self, show: bool = True) -> Optional[Dict[str, Any]]:
        """Inspect the raw data.

        Args:
            show (bool): If True, print the raw data. If False, return it.

        Returns:
            Optional[Dict[str, Any]]: If True, print the raw data and return None. If
                False, return the raw data dictionary.
        """
        raw_data = {
            key: getattr(self, key)
            for key in self.__dict__
            if not key.startswith("_") and key != "log"
        }

        if not show:
            return raw_data

        print(json.dumps(raw_data, indent=2, cls=HumanReadableEncoder))
        return None

    def serialize(self) -> Dict[str, Any]:
        """Serialize the raw data to a dictionary

        Returns:
            Dict[str, Any]: The serialized raw data
        """
        return {key: getattr(self, key) for key in self.__dict__}


@dataclass
class ResultTable:
    """
    A dataclass that holds the table summary of result.
    """

    data: Union[List[Any], pd.DataFrame]
    title: Optional[str] = None

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
    """Base Class for test suite results."""

    result_id: str = None
    name: str = None
    result_type: str = None

    def __str__(self) -> str:
        """May be overridden by subclasses."""
        return self.__class__.__name__

    def to_html(self):
        """Generate HTML representation of the result. Must be overridden by subclasses."""
        raise NotImplementedError

    def log(self):
        """Log the result... Must be overridden by subclasses."""
        raise NotImplementedError

    def show(self):
        """Display the result... May be overridden by subclasses."""
        if hasattr(self, "to_html"):
            display(self.to_html())
        else:
            display(str(self))


@dataclass
class ErrorResult(Result):
    """Result for test suites that fail to load or run properly."""

    name: str = "Failed Test"
    error: Exception = None
    message: str = None

    def __repr__(self) -> str:
        return f'ErrorResult(result_id="{self.result_id}")'

    def to_html(self):
        """Generate HTML that persists in saved notebooks."""
        return f"""
        {StatefulHTMLRenderer.get_base_css()}
        <div class="vm-result">
            <h3 style="color: red;">{self.message}</h3>
            <p>{self.error}</p>
        </div>
        """

    async def log_async(self):
        pass


@dataclass
class TestResult(Result):
    """Test result."""

    name: str = "Test Result"
    ref_id: str = None
    title: Optional[str] = None
    doc: Optional[str] = None
    description: Optional[Union[str, DescriptionFuture]] = None
    metric: Optional[Union[int, float]] = None
    scorer: Optional[List[Union[int, float]]] = None
    tables: Optional[List[ResultTable]] = None
    raw_data: Optional[RawData] = None
    figures: Optional[List[Figure]] = None
    passed: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Union[List[VMInput], VMInput]]] = None
    metadata: Optional[Dict[str, Any]] = None
    _was_description_generated: bool = False
    _unsafe: bool = False
    _client_config_cache: Optional[Any] = None
    _is_scorer_result: bool = False

    def __post_init__(self):
        if self.ref_id is None:
            self.ref_id = str(uuid4())

    def __repr__(self) -> str:
        attrs = [
            attr
            for attr in [
                "doc",
                "description",
                "params",
                "tables",
                "figures",
                "metric",
                "passed",
            ]
            if getattr(self, attr) is not None
            and (
                len(getattr(self, attr)) > 0
                if isinstance(getattr(self, attr), list)
                else True
            )
        ]

        return f'TestResult("{self.result_id}", {", ".join(attrs)})'

    def __getattribute__(self, name):
        # lazy load description if its a DescriptionFuture (generated in background)
        if name == "description":
            description = super().__getattribute__("description")

            if isinstance(description, DescriptionFuture):
                (
                    self.description,
                    self._was_description_generated,
                ) = description.get_description()

        return super().__getattribute__(name)

    @property
    def test_name(self) -> str:
        """Get the test name, using custom title if available."""
        return self.title or test_id_to_name(self.result_id)

    def _get_flat_inputs(self):
        # remove duplicates by `input_id`
        inputs = {}

        for input_or_list in self.inputs.values():
            if isinstance(input_or_list, list):
                inputs.update({input.input_id: input for input in input_or_list})
            else:
                inputs[input_or_list.input_id] = input_or_list

        return list(inputs.values())

    def set_metric(self, values: Union[int, float, List[Union[int, float]]]) -> None:
        """Set the metric value.
        Args:
            values: The metric values to set. Can be int, float, or List[Union[int, float]].
        """
        if isinstance(values, list):
            # Lists should be stored in scorer
            self.scorer = values
            self.metric = None  # Clear metric field when using scorer
        else:
            # Single values should be stored in metric
            self.metric = values
            self.scorer = None  # Clear scorer field when using metric

    def _get_metric_display_value(
        self,
    ) -> Union[int, float, List[Union[int, float]], None]:
        """Get the metric value for display purposes.
        Returns:
            The raw metric value, handling both metric and scorer fields.
        """
        if self.metric is not None:
            return self.metric

        if self.scorer is not None:
            return self.scorer

        return None

    def _get_metric_serialized_value(
        self,
    ) -> Union[int, float, List[Union[int, float]], None]:
        """Get the metric value for API serialization.
        Returns:
            The serialized metric value, handling both metric and scorer fields.
        """
        if self.metric is not None:
            return self.metric

        if self.scorer is not None:
            return self.scorer

        return None

    def _get_metric_type(self) -> Optional[str]:
        """Get the type of metric being stored.
        Returns:
            The metric type identifier or None if no metric is set.
        """
        if self.metric is not None:
            return "unit_metric"

        if self.scorer is not None:
            return "scorer"

        return None

    def add_table(
        self,
        table: Union[ResultTable, pd.DataFrame, List[Dict[str, Any]]],
        title: Optional[str] = None,
    ):
        """Add a new table to the result.

        Args:
            table (Union[ResultTable, pd.DataFrame, List[Dict[str, Any]]]): The table to add.
            title (Optional[str]): The title of the table (can optionally be provided for
                pd.DataFrame and List[Dict[str, Any]] tables).
        """
        if self.tables is None:
            self.tables = []

        if isinstance(table, (pd.DataFrame, list)):
            table = ResultTable(data=table, title=title)

        self.tables.append(table)

    def remove_table(self, index: int):
        """Remove a table from the result by index.

        Args:
            index (int): The index of the table to remove (default is 0).
        """
        if self.tables is None:
            return

        self.tables.pop(index)

    def add_figure(
        self,
        figure: Union[
            matplotlib.figure.Figure,
            go.Figure,
            go.FigureWidget,
            bytes,
            Figure,
        ],
    ) -> None:
        """Add a new figure to the result.

        Args:
            figure: The figure to add. Can be one of:
                - matplotlib.figure.Figure: A matplotlib figure
                - plotly.graph_objs.Figure: A plotly figure
                - plotly.graph_objs.FigureWidget: A plotly figure widget
                - bytes: A PNG image as raw bytes
                - validmind.vm_models.figure.Figure: A ValidMind figure object.

        Returns:
            None.
        """
        if self.figures is None:
            self.figures = []

        if not isinstance(figure, Figure):
            random_tag = str(uuid4())[:4]
            figure = create_figure(
                figure=figure,
                ref_id=self.ref_id,
                key=f"{self.result_id}:{random_tag}",
            )

        if figure.ref_id != self.ref_id:
            figure.ref_id = self.ref_id

        self.figures.append(figure)

    def remove_figure(self, index: int = 0):
        """Remove a figure from the result by index.

        Args:
            index (int): The index of the figure to remove (default is 0).
        """
        if self.figures is None:
            return

        self.figures.pop(index)

    def to_html(self):
        """Generate HTML that persists in saved notebooks."""
        metric_value = self._get_metric_display_value()

        if metric_value is not None and not self.tables and not self.figures:
            return StatefulHTMLRenderer.render_result_header(
                test_name=self.test_name, passed=self.passed, metric=metric_value
            )

        html_parts = [StatefulHTMLRenderer.get_base_css()]

        html_parts.append(
            StatefulHTMLRenderer.render_result_header(
                test_name=self.test_name, passed=self.passed, metric=metric_value
            )
        )

        if self.description:
            html_parts.append(StatefulHTMLRenderer.render_description(self.description))

        if self.params:
            html_parts.append(StatefulHTMLRenderer.render_parameters(self.params))

        if self.tables:
            html_parts.append(tables_to_html(self.tables))

        if self.figures:
            html_parts.append(figures_to_html(self.figures))

        return f'<div class="vm-result">{"".join(html_parts)}</div>'

    @classmethod
    def _get_client_config(cls):
        """Get the client config, loading it if not cached."""
        if cls._client_config_cache is None:
            api_client.reload()
            cls._client_config_cache = api_client.client_config

            if cls._client_config_cache is None:
                raise ValueError(
                    "Failed to load client config: api_client.client_config is None"
                )

            if not hasattr(cls._client_config_cache, "documentation_template"):
                raise ValueError(
                    "Invalid client config: missing documentation_template"
                )

        return cls._client_config_cache

    def check_result_id_exist(self):
        """Check if the result_id exists in any test block across all sections."""
        client_config = self._get_client_config()

        # Iterate through all sections
        for section in client_config.documentation_template["sections"]:
            blocks = section.get("contents", [])
            for block in blocks:
                if (
                    block.get("content_type") == "test"
                    and block.get("content_id") == self.result_id
                ):
                    return

        logger.info(
            f"Test driven block with result_id {self.result_id} does not exist in model's document"
        )

    def _validate_section_id_for_block(
        self, section_id: str, position: Union[int, None] = None
    ):
        """Validate the section_id exits on the template before logging."""
        client_config = self._get_client_config()
        found = False

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
        """Serialize the result for the API."""
        serialized = {
            "test_name": self.result_id,
            "title": self.title,
            "ref_id": self.ref_id,
            "params": self.params,
            "inputs": [_input.input_id for _input in self._get_flat_inputs()],
            "passed": self.passed,
            "summary": [table.serialize() for table in (self.tables or [])],
            "metadata": self.metadata,
        }

        metric_type = self._get_metric_type()
        if metric_type:
            serialized["metric_type"] = metric_type

        return serialized

    async def log_async(
        self,
        section_id: str = None,
        content_id: str = None,
        position: int = None,
        config: Dict[str, bool] = None,
    ):
        # Skip logging for scorers - they should not be saved to the backend
        if self._is_scorer_result:
            return

        tasks = []  # collect tasks to run in parallel (async)

        # Default empty dict if None
        config = config or {}

        tasks.append(
            api_client.alog_test_result(
                result=self.serialize(),
                section_id=section_id,
                position=position,
                config=config,
            )
        )

        if self.metric is not None or self.scorer is not None:
            # metrics are logged as separate entities
            metric_value = self._get_metric_serialized_value()
            metric_type = self._get_metric_type()

            metric_key = self.result_id
            if metric_type == "scorer":
                metric_key = f"{self.result_id}_scorer"

            tasks.append(
                api_client.alog_metric(
                    key=metric_key,
                    value=metric_value,
                    inputs=[input.input_id for input in self._get_flat_inputs()],
                    params=self.params,
                )
            )

        if self.figures:
            batch_size = min(
                len(self.figures), int(os.getenv("VM_FIGURE_MAX_BATCH_SIZE", 20))
            )
            figure_batches = [
                self.figures[i : i + batch_size]
                for i in range(0, len(self.figures), batch_size)
            ]

            async def upload_figures_in_batches():
                for batch in figure_batches:

                    @log_api_operation(
                        operation_name=f"Uploading batch of {len(batch)} figures"
                    )
                    async def process_batch():
                        batch_tasks = [
                            api_client.alog_figure(figure) for figure in batch
                        ]
                        return await asyncio.gather(*batch_tasks)

                    await process_batch()

            tasks.append(upload_figures_in_batches())

        if self.description:
            revision_name = (
                AI_REVISION_NAME
                if self._was_description_generated
                else DEFAULT_REVISION_NAME
            )

            tasks.append(
                update_metadata(
                    content_id=(
                        f"{content_id}::{revision_name}"
                        if content_id
                        else f"test_description:{self.result_id}::{revision_name}"
                    ),
                    text=self.description,
                )
            )

        return await asyncio.gather(*tasks)

    def log(  # noqa: C901
        self,
        section_id: str = None,
        content_id: str = None,
        position: int = None,
        unsafe: bool = False,
        config: Dict[str, bool] = None,
    ):
        """Log the result to ValidMind.

        Args:
            section_id (str): The section ID within the model document to insert the
                test result.
            content_id (str): The content ID to log the result to.
            position (int): The position (index) within the section to insert the test
                result.
            unsafe (bool): If True, log the result even if it contains sensitive data
                i.e. raw data from input datasets.
            config (Dict[str, bool]): Configuration options for displaying the test result.
                Available config options:
                - hideTitle: Hide the title in the document view
                - hideText: Hide the description text in the document view
                - hideParams: Hide the parameters in the document view
                - hideTables: Hide tables in the document view
                - hideFigures: Hide figures in the document view
        """
        if config:
            self.validate_log_config(config)

        self.check_result_id_exist()

        if not unsafe and get_pii_detection_mode() in [
            PIIDetectionMode.TEST_RESULTS,
            PIIDetectionMode.ALL,
        ]:
            for table in self.tables or []:
                scan_df(table.data)

            if self.description:
                scan_text(self.description)

        if section_id:
            self._validate_section_id_for_block(section_id, position)

        run_async(
            self.log_async,
            section_id=section_id,
            content_id=content_id,
            position=position,
            config=config,
        )

    def validate_log_config(self, config: Dict[str, bool]):
        """Validate the configuration options for logging a test result

        Args:
            config (Dict[str, bool]): Configuration options to validate

        Raises:
            InvalidParameterError: If config contains invalid keys or non-boolean values
        """
        valid_keys = {
            "hideTitle",
            "hideText",
            "hideParams",
            "hideTables",
            "hideFigures",
        }
        invalid_keys = set(config.keys()) - valid_keys
        if invalid_keys:
            raise InvalidParameterError(
                f"Invalid config keys: {', '.join(invalid_keys)}. "
                f"Valid keys are: {', '.join(valid_keys)}"
            )

        # Ensure all values are boolean
        non_bool_keys = [
            key for key, value in config.items() if not isinstance(value, bool)
        ]
        if non_bool_keys:
            raise InvalidParameterError(
                f"Values for config keys must be boolean. Non-boolean values found for keys: {', '.join(non_bool_keys)}"
            )


@dataclass
class TextGenerationResult(Result):
    """Test result."""

    name: str = "Text Generation Result"
    ref_id: str = None
    title: Optional[str] = None
    doc: Optional[str] = None
    description: Optional[Union[str, DescriptionFuture]] = None
    params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    _was_description_generated: bool = False

    def __post_init__(self):
        if self.ref_id is None:
            self.ref_id = str(uuid4())

    def __repr__(self) -> str:
        attrs = [
            attr
            for attr in [
                "doc",
                "description",
                "params",
            ]
            if getattr(self, attr) is not None
            and (
                len(getattr(self, attr)) > 0
                if isinstance(getattr(self, attr), list)
                else True
            )
        ]

        return f'TextGenerationResult("{self.result_id}", {", ".join(attrs)})'

    def __getattribute__(self, name):
        # lazy load description if its a DescriptionFuture (generated in background)
        if name == "description":
            description = super().__getattribute__("description")

            if isinstance(description, DescriptionFuture):
                self._was_description_generated = True
                self.description = description.get_description()

        return super().__getattribute__(name)

    @property
    def test_name(self) -> str:
        """Get the test name, using custom title if available."""
        return self.title or test_id_to_name(self.result_id)

    def to_html(self):
        """Generate HTML that persists in saved notebooks."""
        html_parts = [StatefulHTMLRenderer.get_base_css()]

        html_parts.append(
            StatefulHTMLRenderer.render_result_header(
                test_name=self.test_name, passed=None
            )
        )

        if self.description:
            html_parts.append(StatefulHTMLRenderer.render_description(self.description))

        if self.params:
            html_parts.append(StatefulHTMLRenderer.render_parameters(self.params))

        return f'<div class="vm-result">{"".join(html_parts)}</div>'

    def serialize(self):
        """Serialize the result for the API."""
        return {
            "test_name": self.result_id,
            "title": self.title,
            "ref_id": self.ref_id,
            "params": self.params,
            "metadata": self.metadata,
        }

    async def log_async(
        self,
        content_id: str = None,
    ):
        return await asyncio.gather(
            update_metadata(
                content_id=f"{content_id}",
                text=self.description,
            )
        )

    def log(
        self,
        content_id: str = None,
    ):
        """Log the result to ValidMind.

        Args:
            content_id (str): The content ID to log the result to.
        """
        if self.description:
            try:
                from .pii_filter import check_text_for_pii

                check_text_for_pii(self.description, raise_on_detection=True)
            except ImportError:
                logger.debug(
                    "PII detection not available - skipping PII check for description"
                )
            except ValueError:
                # Re-raise PII detection errors
                raise
            except Exception as e:
                logger.warning(f"PII detection failed for description: {e}")

        run_async(
            self.log_async,
            content_id=content_id,
        )
