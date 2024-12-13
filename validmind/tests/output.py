# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from validmind.vm_models.figure import (
    Figure,
    is_matplotlib_figure,
    is_plotly_figure,
    is_png_image,
)
from validmind.vm_models.result import RawData, ResultTable, TestResult


class OutputHandler(ABC):
    """Base class for handling different types of test outputs"""

    @abstractmethod
    def can_handle(self, item: Any) -> bool:
        """Check if this handler can process the given item"""
        pass

    @abstractmethod
    def process(self, item: Any, result: TestResult) -> None:
        """Process the item and update the TestResult"""
        pass


class BooleanOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, (bool, np.bool_))

    def process(self, item: Any, result: TestResult) -> None:
        if result.passed is not None:
            raise ValueError("Test returned more than one boolean value")
        result.passed = bool(item)


class MetricOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, (int, float))

    def process(self, item: Any, result: TestResult) -> None:
        if result.metric is not None:
            raise ValueError("Only one unit metric may be returned per test.")
        result.metric = item


class FigureOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return (
            isinstance(item, Figure)
            or is_matplotlib_figure(item)
            or is_plotly_figure(item)
            or is_png_image(item)
        )

    def process(self, item: Any, result: TestResult) -> None:
        if isinstance(item, Figure):
            result.add_figure(item)
        else:
            random_id = str(uuid4())[:4]
            result.add_figure(
                Figure(
                    key=f"{result.result_id}:{random_id}",
                    figure=item,
                    ref_id=result.ref_id,
                )
            )


class TableOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, (list, pd.DataFrame, dict, ResultTable))

    def process(
        self,
        item: Union[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any], ResultTable],
        result: TestResult,
    ) -> None:
        tables = item if isinstance(item, dict) else {"": item}

        for table_name, table_data in tables.items():
            # if already a ResultTable, add it directly
            if isinstance(table_data, ResultTable):
                result.add_table(table_data)
                continue

            if not isinstance(table_data, (list, pd.DataFrame)):
                raise ValueError(
                    "Invalid table format: must be a list of dictionaries or a DataFrame"
                )

            if isinstance(table_data, list):
                table_data = pd.DataFrame(table_data)

            result.add_table(ResultTable(data=table_data, title=table_name or None))


class RawDataOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, RawData)

    def process(self, item: Any, result: TestResult) -> None:
        result.raw_data = item


def process_output(item: Any, result: TestResult) -> None:
    """Process a single test output item and update the TestResult."""
    handlers = [
        BooleanOutputHandler(),
        MetricOutputHandler(),
        FigureOutputHandler(),
        TableOutputHandler(),
        RawDataOutputHandler(),
    ]

    for handler in handlers:
        if handler.can_handle(item):
            handler.process(item, result)
            return

    raise ValueError(f"Invalid test output type: {type(item)}")
