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
    from deepeval.metrics import BiasMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for Bias. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "Bias", "deepeval")
@tasks("llm")
def Bias(
    dataset: VMDataset,
    threshold: float = 0.5,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Detects bias in LLM outputs using deepeval's BiasMetric.

    This scorer evaluates whether an LLM's output contains biased opinions (e.g.,
    gender, racial, political). It returns a list of dictionaries per row with the
    bias score and the reason behind the evaluation.

    Args:
        dataset: The dataset containing input prompts and LLM outputs
        threshold: Maximum passing threshold for bias (default: 0.5)
        input_column: Column name for the input prompts (default: "input")
        actual_output_column: Column name for the model outputs (default: "actual_output")
        strict_mode: If True, enforces a binary score (0 for perfect, 1 otherwise)

    Returns:
        List[Dict[str, Any]]: Per-row bias scores and reasons as a list of dictionaries.
        Each dictionary contains:
        - "score": float - The bias score (0.0 to 1.0)
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

    metric = BiasMetric(
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

        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])

        # Extract score and reason from the metric result
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")

        results.append({"score": score, "reason": reason})

    return results
