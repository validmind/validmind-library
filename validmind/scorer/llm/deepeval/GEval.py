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
    from deepeval.metrics import GEval as geval
    from deepeval.metrics.g_eval.utils import Rubric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for GEval. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "GEval", "deepeval")
@tasks("llm")
def GEval(
    dataset: VMDataset,
    metric_name: str,
    criteria: str,
    evaluation_steps: List[str] = [],
    rubrics: List[Dict[str, Any]] = [],
    strict_mode: bool = False,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Detects evaluation criteria in LLM outputs using deepeval's GEval metric.

    This scorer evaluates whether an LLM's output contains the specified evaluation criteria. It uses the GEval framework
    (https://arxiv.org/pdf/2303.16634.pdf) to assess outputs against defined criteria and rubrics. The scorer processes each row
    in the dataset and returns evaluation scores and explanations.

    Args:
        dataset (VMDataset): Dataset containing input prompts and LLM outputs to evaluate
        metric_name (str): Name of the GEval metric to use for evaluation
        criteria (str): Evaluation criteria to assess the outputs against
        evaluation_steps (List[str], optional): Specific steps to follow during evaluation. Defaults to empty list.
        rubrics (List[Dict[str, Any]], optional): List of rubric dictionaries defining evaluation criteria. Each rubric should
            contain score and description. Defaults to empty list.
        strict_mode (bool, optional): If True, enforces binary scoring (0 or 1). If False, allows fractional scores.
            Defaults to False.
        threshold (float, optional): Minimum score threshold for considering an evaluation successful. Range 0.0-1.0.
            Defaults to 0.5.

    Returns:
        List[Dict[str, Any]]: List of evaluation results per dataset row. Each dictionary contains:
            - score (float): Evaluation score between 0.0 and 1.0 (or 0/1 if strict_mode=True)
            - reason (str): Detailed explanation of the evaluation and score assignment

    Raises:
        ValueError: If required input, actual_output or expected_output columns are missing from dataset
        MissingDependencyError: If the required deepeval package is not installed

    Example:
        results = GEval(
            dataset=my_dataset,
            metric_name="response_quality",
            criteria="Response should be clear, accurate and well-structured",
            rubrics=[{"score": 1, "description": "Perfect response"},
                    {"score": 0, "description": "Poor response"}],
            strict_mode=True
        )
    """

    # Validate required columns exist in dataset
    if "input" not in dataset._df.columns:
        raise ValueError(
            f"Input column 'input' not found in dataset. Available columns: {dataset._df.columns.tolist()}"
        )

    if "actual_output" not in dataset._df.columns:
        raise ValueError(
            f"Actual output column 'actual_output' not found in dataset. Available columns: {dataset._df.columns.tolist()}"
        )
    if "expected_output" not in dataset._df.columns:
        raise ValueError(
            f"Expected output column 'expected_output' not found in dataset. Available columns: {dataset._df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    evaluation_params = {
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    }

    rubrics_list = []
    if rubrics:
        rubrics_list = [Rubric(**rubric) for rubric in rubrics]

    metric = geval(
        name=metric_name,
        criteria=criteria,
        evaluation_params=evaluation_params,
        model=model,
        evaluation_steps=evaluation_steps if evaluation_steps else None,
        rubric=rubrics_list if rubrics_list else None,
        strict_mode=strict_mode,
        verbose_mode=False,
        threshold=threshold,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset._df.iterrows():
        test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"],
        )

        result = metric.measure(test_case)
        metric_name = metric_name.replace(" ", "_")
        results.append({f"{metric_name}_score": result})

    return results
