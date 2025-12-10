# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List, Optional

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval import evaluate
    from deepeval.metrics import SummarizationMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for Summarization. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "Summarization", "deepeval")
@tasks("llm")
def Summarization(
    dataset: VMDataset,
    threshold: float = 0.5,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
    assessment_questions: Optional[List[str]] = None,
    n: int = 5,
    truths_extraction_limit: Optional[int] = None,
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates summary quality using deepeval's SummarizationMetric.

    The metric generates or uses provided close-ended questions to assess if the
    summary is factually aligned with and sufficiently covers the source text.

    Args:
        dataset: Dataset containing original text and generated summary
        threshold: Minimum passing threshold (default: 0.5)
        input_column: Column name for the original text (default: "input")
        actual_output_column: Column for the generated summary (default: "actual_output")
        assessment_questions: Optional list of yes/no questions to assess the summary
        n: Number of assessment questions to generate when not provided (default: 5)
        truths_extraction_limit: Optional cap for number of truths extracted from input
        strict_mode: If True, enforces a binary score (0 for perfect, 1 otherwise)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """

    # Validate required columns exist in dataset
    missing_columns: List[str] = []
    for col in [input_column, actual_output_column]:
        if col not in dataset.df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset.df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    # Build metric with optional parameters
    metric_kwargs: Dict[str, Any] = dict(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )
    if assessment_questions is not None:
        metric_kwargs["assessment_questions"] = assessment_questions
    else:
        metric_kwargs["n"] = n
    if truths_extraction_limit is not None:
        metric_kwargs["truths_extraction_limit"] = truths_extraction_limit

    metric = SummarizationMetric(**metric_kwargs)

    results: List[Dict[str, Any]] = []
    for _, row in dataset.df.iterrows():
        input_value = row[input_column]
        actual_output_value = row[actual_output_column]

        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
