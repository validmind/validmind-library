# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Class for storing ValidMind metric objects and associated
data for display and reporting purposes
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Union

import pandas as pd

from ...ai.test_descriptions import get_description_metadata
from ...errors import MissingCacheResultsArgumentsError
from ..figure import Figure
from .metric_result import MetricResult
from .result_wrapper import MetricResultWrapper
from .test import Test


@dataclass
class Metric(Test):
    """
    Metric objects track the schema supported by the ValidMind API
    """

    # Class Variables
    test_type: ClassVar[str] = "Metric"

    type: ClassVar[str] = ""  # type of metric: "training", "evaluation", etc.
    scope: ClassVar[str] = ""  # scope of metric: "training_dataset"
    value_formatter: ClassVar[Optional[str]] = None  # "records" or "key_values"

    # Instance Variables
    result: MetricResultWrapper = None  # populated by cache_results() method

    @abstractmethod
    def summary(self, metric_value: Optional[Union[dict, list, pd.DataFrame]] = None):
        """
        Return the metric summary. Should be overridden by subclasses. Defaults to None.
        The metric summary allows renderers (e.g. Word and ValidMind UI) to display a
        short summary of the metric results.

        We return None here because the metric summary is optional.
        """
        return None

    def cache_results(
        self,
        metric_value: Optional[Union[dict, list, pd.DataFrame]] = None,
        figures: Optional[List[Figure]] = None,
    ):
        """
        Cache the results of the metric calculation and do any post-processing if needed

        Args:
            metric_value (Union[dict, list, pd.DataFrame]): The value of the metric
            figures (Optional[object]): Any figures to attach to the test suite result

        Returns:
            TestSuiteResult: The test suite result object
        """
        if metric_value is None and figures is None:
            raise MissingCacheResultsArgumentsError(
                "Metric must provide a metric value or figures to cache_results"
            )

        metric = MetricResult(
            key=self.test_id,
            ref_id=self._ref_id,
            value=metric_value if metric_value is not None else {},
            value_formatter=self.value_formatter,
            summary=self.summary(metric_value),
        )

        self.result = MetricResultWrapper(
            result_id=self.test_id,
            result_description=self.description(),
            result_metadata=[
                (
                    get_description_metadata(
                        test_id=self.test_id,
                        default_description=self.description(),
                        summary=metric.serialize()["summary"],
                        figures=figures,
                        should_generate=self.generate_description,
                    )
                )
            ],
            metric=metric,
            figures=figures,
            inputs=self.get_accessed_inputs(),
            output_template=self.output_template,
        )

        return self.result
