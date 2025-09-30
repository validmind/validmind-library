# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from validmind.utils import is_html, md_to_html
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
        return isinstance(item, (list, pd.DataFrame, dict, ResultTable, tuple))

    def _convert_simple_type(self, data: Any) -> pd.DataFrame:
        """Convert a simple data type to a DataFrame."""
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif data is None:
            return pd.DataFrame()
        else:
            raise ValueError(f"Cannot convert {type(data)} to DataFrame")

    def _convert_list(self, data_list: List) -> pd.DataFrame:
        """Convert a list to a DataFrame."""
        if not data_list:
            return pd.DataFrame()

        try:
            return pd.DataFrame(data_list)
        except Exception as e:
            # If conversion fails, try to handle common cases
            if all(
                isinstance(item, (int, float, str, bool, type(None)))
                for item in data_list
            ):
                return pd.DataFrame({"Values": data_list})
            else:
                raise ValueError(f"Could not convert list to DataFrame: {e}")

    def _convert_to_dataframe(self, table_data: Any) -> pd.DataFrame:
        """Convert various data types to a pandas DataFrame."""
        # Handle special cases by type
        if isinstance(table_data, pd.DataFrame):
            return table_data
        elif isinstance(table_data, (dict, str, type(None))):
            return self._convert_simple_type(table_data)
        elif isinstance(table_data, tuple):
            return self._convert_list(list(table_data))
        elif isinstance(table_data, list):
            return self._convert_list(table_data)
        else:
            # If we reach here, we don't know how to handle this type
            raise ValueError(
                f"Invalid table format: must be a list of dictionaries or a DataFrame, got {type(table_data)}"
            )

    def process(
        self,
        item: Union[
            List[Dict[str, Any]], pd.DataFrame, Dict[str, Any], ResultTable, str, tuple
        ],
        result: TestResult,
    ) -> None:
        # Convert to a dictionary of tables if not already
        tables = item if isinstance(item, dict) else {"": item}

        for table_name, table_data in tables.items():
            # If already a ResultTable, add it directly
            if isinstance(table_data, ResultTable):
                result.add_table(table_data)
                continue

            # Convert the data to a DataFrame using our helper method
            df = self._convert_to_dataframe(table_data)

            # Add the resulting DataFrame as a table to the resul
            result.add_table(ResultTable(data=df, title=table_name or None))


class RawDataOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, RawData)

    def process(self, item: Any, result: TestResult) -> None:
        result.raw_data = item


class StringOutputHandler(OutputHandler):
    def can_handle(self, item: Any) -> bool:
        return isinstance(item, str)

    def process(self, item: Any, result: TestResult) -> None:
        if not is_html(item):
            item = md_to_html(item, mathml=True)

        result.description = item


class ScorerOutputHandler(OutputHandler):
    """Handler for scorer outputs that should not be logged to backend"""

    def can_handle(self, item: Any) -> bool:
        # This handler is only called when we've already determined it's a scorer
        # based on the _is_scorer marker on the test function
        return True

    def process(self, item: Any, result: TestResult) -> None:
        # For scorers, we just store the raw output without special processing
        # The output will be used by the calling code (e.g., assign_scores)
        # but won't be logged to the backend
        result.raw_data = RawData(scorer_output=item)


def process_output(
    item: Any, result: TestResult, test_func: Optional[Callable] = None
) -> None:
    """Process a single test output item and update the TestResult."""
    handlers = [
        BooleanOutputHandler(),
        FigureOutputHandler(),
        TableOutputHandler(),
        RawDataOutputHandler(),
        StringOutputHandler(),
        # Unit metrics should be processed last
        MetricOutputHandler(),
    ]

    # Check if this is a scorer first by looking for the _is_scorer marker
    if test_func and hasattr(test_func, "_is_scorer") and test_func._is_scorer:
        # For scorers, handle the output specially
        scorer_handler = ScorerOutputHandler()
        scorer_handler._result = result
        if scorer_handler.can_handle(item):
            scorer_handler.process(item, result)
            return

    for handler in handlers:
        if handler.can_handle(item):
            handler.process(item, result)
            return

    raise ValueError(f"Invalid test output type: {type(item)}")
