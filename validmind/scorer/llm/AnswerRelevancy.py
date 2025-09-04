# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset
from validmind.vm_models.result.result import RowMetricValues


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "AnswerRelevancy", "deepeval")
@tasks("llm")
def AnswerRelevancy(
    dataset: VMDataset,
    threshold: float = 0.8,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
) -> Dict[str, Any]:

    # Validate required columns exist in dataset
    if input_column not in dataset.df.columns:
        raise ValueError(
            f"Input column '{input_column}' not found in dataset. Available columns: {dataset.df.columns.tolist()}"
        )

    if actual_output_column not in dataset.df.columns:
        raise ValueError(
            f"Actual output column '{actual_output_column}' not found in dataset. Available columns: {dataset.df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    metric = AnswerRelevancyMetric(
        threshold=threshold, model=model, include_reason=True, verbose_mode=False
    )
    results = []
    for _, test_case in dataset.df.iterrows():
        input = test_case["input"]
        actual_output = test_case["actual_output"]

        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
        )
        result = evaluate(test_cases=[test_case], metrics=[metric])
        print(result.test_results[0].metrics_data[0].score)
        results.append(result.test_results[0].metrics_data[0].score)

    return RowMetricValues(results)
