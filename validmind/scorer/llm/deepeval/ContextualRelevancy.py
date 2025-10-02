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
    from deepeval.metrics import ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for ContextualRelevancyMetric. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "ContextualRelevancy", "deepeval")
@tasks("llm")
def ContextualRelevancy(
    dataset: VMDataset,
    threshold: float = 0.5,
    input_column: str = "input",
    expected_output_column: str = "expected_output",
    retrieval_context_column: str = "retrieval_context",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates RAG retriever relevancy using deepeval's ContextualRelevancyMetric.

    This metric checks whether statements in the retrieved context are relevant to the
    query-only input. Returns per-row score and reason.

    Args:
        dataset: Dataset containing query, expected_output, and retrieval_context
        threshold: Minimum passing threshold (default: 0.5)
        input_column: Column name for the query-only input (default: "input")
        expected_output_column: Column for the reference output (default: "expected_output")
        retrieval_context_column: Column with ranked retrieved nodes list (default: "retrieval_context")
        strict_mode: If True, enforces a binary score (0 for perfect, 1 otherwise)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """

    # Validate required columns
    missing_columns: List[str] = []
    for col in [input_column, expected_output_column, retrieval_context_column]:
        if col not in dataset.df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset.df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    metric = ContextualRelevancyMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset.df.iterrows():
        input_value = row[input_column]
        expected_output_value = row[expected_output_column]
        retrieval_context_value = (
            [row[retrieval_context_column]]
            if not isinstance(row[retrieval_context_column], list)
            else row[retrieval_context_column]
        )

        # Ensure retrieval_context is a list of strings
        if not isinstance(retrieval_context_value, list):
            raise ValueError(
                f"Value in '{retrieval_context_column}' must be a list of strings; got {type(retrieval_context_value)}"
            )

        test_case = LLMTestCase(
            input=input_value,
            expected_output=expected_output_value,
            retrieval_context=retrieval_context_value,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
