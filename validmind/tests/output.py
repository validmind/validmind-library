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
        return isinstance(item, (list, pd.DataFrame, dict, ResultTable, str, tuple))

    def process(
        self,
        item: Union[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any], ResultTable, str, tuple],
        result: TestResult,
    ) -> None:
        tables = item if isinstance(item, dict) else {"": item}

        for table_name, table_data in tables.items():
            # if already a ResultTable, add it directly
            if isinstance(table_data, ResultTable):
                result.add_table(table_data)
                continue

            if not isinstance(table_data, (list, pd.DataFrame)):
                # Try to convert to a valid format if possible
                if isinstance(table_data, dict):
                    # Convert dict to a single-row DataFrame
                    table_data = pd.DataFrame([table_data])
                elif isinstance(table_data, str):
                    # Convert string to a single-cell DataFrame
                    table_data = pd.DataFrame({'Value': [table_data]})
                elif isinstance(table_data, tuple):
                    # Convert tuple to a list, which will be converted to a DataFrame later
                    table_data = list(table_data)
                elif table_data is None:
                    # Skip None values
                    continue
                else:
                    # If conversion isn't possible, raise a more detailed error
                    raise ValueError(
                        f"Invalid table format: must be a list of dictionaries or a DataFrame, got {type(table_data)}"
                    )

            if isinstance(table_data, list):
                if len(table_data) > 0:
                    # Try to convert to DataFrame, handling potential conversion errors
                    try:
                        table_data = pd.DataFrame(table_data)
                    except Exception as e:
                        # If conversion fails, try to handle common cases
                        if all(isinstance(item, (int, float, str, bool, type(None))) for item in table_data):
                            # For simple types, create a single column DataFrame
                            table_data = pd.DataFrame({'Values': table_data})
                        else:
                            # If we can't handle it, raise a more informative error
                            raise ValueError(f"Could not convert list to DataFrame: {e}")
                else:
                    # Handle empty list case
                    table_data = pd.DataFrame()

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
