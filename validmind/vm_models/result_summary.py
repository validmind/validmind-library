# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from dataclasses import dataclass
from typing import Any, List, Union

import pandas as pd


@dataclass
class ResultTableMetadata:
    """
    A dataclass that holds the metadata of a table summary
    """

    title: str


@dataclass
class ResultTable:
    """
    A dataclass that holds the table summary of result
    """

    data: Union[List[Any], pd.DataFrame]
    type: str = "table"
    metadata: ResultTableMetadata = None

    def serialize(self, as_df=False):
        """
        Serializes the Figure to a dictionary so it can be sent to the API.

        This method accepts as_df parameter to return the data as a DataFrame
        if we're returning the data to R.
        """
        table_result = {
            "type": self.type,
        }

        # Convert to a DataFrame so that we can round the values in a standard way
        table_df = pd.DataFrame(self.data) if isinstance(self.data, list) else self.data
        table_df = table_df.round(4)

        if as_df:
            table_result["data"] = table_df
        else:
            table_result["data"] = table_df.to_dict(orient="records")

        if self.metadata is not None:
            table_result["metadata"] = vars(self.metadata)

        return table_result


@dataclass()
class ResultSummary:
    """
    A dataclass that holds the summary of a metric or threshold test results
    """

    results: List[ResultTable]  # TBD other types of results

    def add_result(self, result: ResultTable):
        """
        Adds a result to the list of results
        """
        if self.results is None:
            self.results = []
        self.results.append(result)

    def serialize(self, as_df=False):
        """
        Serializes the ResultSummary to a list of results
        """
        return [result.serialize(as_df) for result in self.results]
