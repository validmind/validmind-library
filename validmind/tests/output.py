# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from validmind.vm_models import VMDataset, VMInput
from validmind.vm_models.figure import (
    Figure,
    is_matplotlib_figure,
    is_plotly_figure,
    is_png_image,
)
from validmind.vm_models.result import ResultTable, TestResult


def check_for_sensitive_data(data: pd.DataFrame, inputs: Dict[str, VMInput]):
    """Check if a table contains raw data from input datasets"""
    dataset_columns = {
        col: len(input_obj.df)
        for input_obj in inputs.values()
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

    def process(
        self,
        item: Union[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]],
        result: TestResult,
    ) -> None:
        tables = item if isinstance(item, dict) else {"": item}

        for table_name, table_data in tables.items():
            if not isinstance(table_data, (list, pd.DataFrame)):
                raise ValueError(
                    f"Invalid table format: must be a list of dictionaries or a DataFrame"
                )

            if isinstance(table_data, list):
                table_data = pd.DataFrame(table_data)

            check_for_sensitive_data(table_data, result.inputs)
            result.add_table(ResultTable(data=table_data, title=table_name or None))


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
