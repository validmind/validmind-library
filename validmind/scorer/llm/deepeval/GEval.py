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
    evaluation_params: Dict[LLMTestCaseParams, str],
    evaluation_steps: List[str] = [],
    rubric: List[Rubric] = None,
    strict_mode: bool = True,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Detects evaluation criteria in LLM outputs using deepeval's GEval metric.

    This scorer evaluates whether an LLM's output contains the specified evaluation criteria. It uses the GEval framework
    (https://arxiv.org/pdf/2303.16634.pdf) to assess outputs against defined criteria and rubrics. The scorer processes each row
    in the dataset and returns evaluation scores and explanations.

    The GEval metric requires the dataset to contain 'input', 'actual_output', and 'expected_output' columns. The 'input' column
    should contain the prompts given to the LLM, 'actual_output' should contain the LLM's responses, and 'expected_output' should
    contain the expected/reference responses.

    Args:
        dataset (VMDataset): Dataset containing input prompts and LLM outputs to evaluate. Must have columns:
            - input: Prompts given to the LLM
            - actual_output: LLM's responses to evaluate
            - expected_output: Expected/reference responses
        metric_name (str): Name of the GEval metric to use for evaluation (e.g., "response_quality", "factual_accuracy")
        criteria (str): Evaluation criteria to assess the outputs against. Should clearly specify what aspects to evaluate.
        evaluation_steps (List[str], optional): Step-by-step instructions for evaluation. Each step should be a clear directive.
            Defaults to empty list.
        rubric (List[Rubric], optional): List of Rubric objects defining evaluation criteria. Each rubric should specify
            scoring criteria and descriptions. Defaults to None.
        strict_mode (bool, optional): If True, enforces binary scoring (0 or 1). If False, allows fractional scores.
            Defaults to True.
        threshold (float, optional): Minimum score threshold for considering an evaluation successful. Range 0.0-1.0.
            Defaults to 0.5.

    Returns:
        List[Dict[str, Any]]: List of evaluation results per dataset row. Each dictionary contains:
            - score (float): Evaluation score between 0.0 and 1.0 (or 0/1 if strict_mode=True)
            - reason (str): Detailed explanation of the evaluation and score assignment
            - metric_name (str): Name of the metric used for evaluation
            - criteria (str): Evaluation criteria used
            - threshold (float): Score threshold used

    Raises:
        ValueError: If required columns ('input', 'actual_output', 'expected_output') are missing from dataset
        MissingDependencyError: If the required deepeval package is not installed

    Example:
        >>> results = GEval(
        ...     dataset=my_dataset,
        ...     metric_name="response_quality",
        ...     criteria="Response should be clear, accurate and well-structured",
        ...     rubric=[
        ...         Rubric(score=1, description="Perfect response meeting all criteria"),
        ...         Rubric(score=0, description="Response fails to meet criteria")
        ...     ],
        ...     strict_mode=True,
        ...     threshold=0.7
        ... )
    """
    _, model = get_client_and_model()

    results: List[Dict[str, Any]] = []
    evaluation_params_dict = {
        value: key.value for key, value in evaluation_params.items()
    }
    df = dataset._df.copy(deep=True)
    # Check if all evaluation parameter columns exist in dataframe
    missing_cols = [
        col for col in evaluation_params_dict.keys() if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Required columns missing from dataset: {missing_cols}")
    df = df.rename(columns=evaluation_params_dict)
    columns = df.columns.tolist()

    for _, row in df.iterrows():
        test_case_dict = {
            key: row[key.value]
            for key in evaluation_params.keys()
            if key.value in columns and row[key.value] is not None
        }
        test_case = LLMTestCase(
            **{key.value: row[key.value] for key in test_case_dict.keys()}
        )

        # evaluation_params = []
        # for param in test_case_dict.keys():
        #     evaluation_params.append(getattr(LLMTestCaseParams, param.upper()))

        metric = geval(
            name=metric_name,
            criteria=criteria,
            evaluation_params=list(test_case_dict.keys()),
            model=model,
            evaluation_steps=evaluation_steps if evaluation_steps else None,
            rubric=rubric if rubric else None,
            strict_mode=strict_mode,
            verbose_mode=False,
            threshold=threshold,
        )
        metric.measure(test_case)
        metric_name = metric_name.replace(" ", "_")
        results.append(
            {
                f"{metric_name}_score": metric.score,
                f"{metric_name}_reason": metric.reason,
                f"{metric_name}_criteria": criteria,
            }
        )

    return results
