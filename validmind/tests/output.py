# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from validmind.vm_models.figure import (
    Figure,
    is_matplotlib_figure,
    is_plotly_figure,
    is_png_image,
)
from validmind.vm_models.result import ResultTable, TestResult


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
            result.add_figure(
                Figure(
                    key=f"{result.result_id}:{len(result.figures) + 1}",
                    figure=item,
                    ref_id=result.ref_id,
                )
            )


class TableOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, (list, pd.DataFrame, dict))

    def process(self, item: Any, result: TestResult) -> None:
        if isinstance(item, (list, pd.DataFrame)):
            result.add_table(ResultTable(data=item))
        else:  # dict case
            for table_name, table in item.items():
                if not isinstance(table, (list, pd.DataFrame)):
                    raise ValueError(
                        f"Invalid table format: {table_name} must be a list or DataFrame"
                    )
                result.add_table(ResultTable(data=table, title=table_name))


def process_output(item: Any, result: TestResult) -> None:
    """Process a single test output item and update the TestResult."""
    handlers = [
        BooleanOutputHandler(),
        MetricOutputHandler(),
        FigureOutputHandler(),
        TableOutputHandler(),
    ]

    for handler in handlers:
        if handler.can_handle(item):
            handler.process(item, result)
            return

    raise ValueError(f"Invalid test output type: {type(item)}")
