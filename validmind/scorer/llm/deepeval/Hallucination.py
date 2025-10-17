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
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for Hallucination. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "Hallucination", "deepeval")
@tasks("llm")
def Hallucination(
    dataset: VMDataset,
    threshold: float = 0.5,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
    context_column: str = "context",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Detects hallucinations in LLM outputs using deepeval's HallucinationMetric.

    The metric checks whether the actual output contradicts the provided context,
    treating the context as ground truth. Returns per-row score and reason.

    Args:
        dataset: Dataset containing input, actual_output, and context
        threshold: Maximum passing threshold (default: 0.5)
        input_column: Column name for the input (default: "input")
        actual_output_column: Column for the model output (default: "actual_output")
        context_column: Column with context list (default: "context")
        strict_mode: If True, enforces a binary score (0 for perfect, 1 otherwise)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """

    # Validate required columns exist in dataset
    missing_columns: List[str] = []
    for col in [input_column, actual_output_column, context_column]:
        if col not in dataset.df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset.df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    metric = HallucinationMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset.df.iterrows():
        input_value = row[input_column]
        actual_output_value = row[actual_output_column]
        context_value = (
            [row[context_column]]
            if not isinstance(row[context_column], list)
            else row[context_column]
        )

        # Ensure context is a list of strings
        if not isinstance(context_value, list):
            raise ValueError(
                f"Value in '{context_column}' must be a list of strings; got {type(context_value)}"
            )

        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
            context=context_value,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
