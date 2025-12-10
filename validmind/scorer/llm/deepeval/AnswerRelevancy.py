# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for AnswerRelevancy. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "AnswerRelevancy", "deepeval")
@tasks("llm")
def AnswerRelevancy(
    dataset: VMDataset,
    threshold: float = 0.8,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
) -> List[Dict[str, Any]]:
    """Calculates answer relevancy scores with explanations for LLM responses.

    This scorer evaluates how relevant an LLM's answer is to the given input question.
    It returns a list of dictionaries, where each dictionary contains both the relevancy
    score and the reasoning behind the score for each row in the dataset.

    Args:
        dataset: The dataset containing input questions and LLM responses
        threshold: The threshold for determining relevancy (default: 0.8)
        input_column: Name of the column containing input questions (default: "input")
        actual_output_column: Name of the column containing LLM responses (default: "actual_output")

    Returns:
        List[Dict[str, Any]]: Per-row relevancy scores and reasons as a list of dictionaries.
        Each dictionary contains:
        - "score": float - The relevancy score (0.0 to 1.0)
        - "reason": str - Explanation of why the score was assigned

    Raises:
        ValueError: If required columns are not found in the dataset
    """

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
        input = test_case[input_column]
        actual_output = test_case[actual_output_column]

        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
        )
        result = evaluate(test_cases=[test_case], metrics=[metric])

        # Extract score and reason from the metric result
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")

        # Create dictionary with score and reason
        results.append({"score": score, "reason": reason})

    return results
