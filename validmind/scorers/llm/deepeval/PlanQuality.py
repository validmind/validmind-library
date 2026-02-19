# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import uuid
from typing import Any, Dict, List, Optional

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models import VMModel
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval.metrics import PlanQualityMetric
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for PlanQuality. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


@scorer()
@tags("llm", "deepeval", "agent_evaluation", "reasoning_layer", "agentic")
@tasks("llm")
def PlanQuality(
    dataset: VMDataset,
    model: Optional[VMModel] = None,
    threshold: float = 0.7,
    input_column: str = "input",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluates the quality of an agent's generated plan for a given input using DeepEval's PlanQualityMetric.

    This scorer measures whether each plan is logical, complete, and efficient for the agent's assigned task.
    It is designed for agentic LLM trace evaluation: when a ValidMind model is provided, the agent is executed
    for each dataset row within DeepEval's evals_iterator, enabling the metric to access the captured trace data.
    If no model is provided, evaluation operates on pre-generated data present in the dataset, which may limit accuracy.

    Args:
        dataset (VMDataset): Dataset with required agent input and plan data.
        model (Optional[VMModel]): ValidMind agent model with a 'predict_fn' method. Ensures evaluation is trace-based.
        threshold (float): Minimum score required to pass (default: 0.7).
        input_column (str): Name of the dataset column containing the primary agent input (default: "input").
        strict_mode (bool): If True, restricts scores to 0 or 1 (binary evaluation).

    Returns:
        List[Dict[str, Any]]: For each row, a dictionary with:
            - "score" (float): Quality score assigned by the metric.
            - "reason" (str): Explanation from the metric regarding the plan quality.

    Raises:
        ValueError: If required columns are not present in the dataset.
        MissingDependencyError: If the 'deepeval' package or its required objects are not installed.

    Example:
        >>> results = PlanQuality(dataset=my_dataset, model=my_agent)
        >>> print(results[0]["score"], results[0]["reason"])

    Test Purpose:
        Validates agentic reasoning quality by ensuring generated plans are robust, actionable, and well-aligned
        with intended tasks. This strengthens trust in agent-based LLM workflows by systematically assessing
        plan structure and effectiveness.

    Interpretation:
        - High "score" values indicate strong plan quality as judged by the LLM.
        - The "reason" field contains diagnostic information about what contributed to the score.
    """
    # Trace-based path: run agent inside evals_iterator so PlanQualityMetric sees the trace
    if model is None or not hasattr(model, "predict_fn") or model.predict_fn is None:
        raise ValueError(
            "PlanQuality requires a `model` with a callable `predict_fn` for trace-based evaluation."
        )
    try:
        from deepeval.dataset import EvaluationDataset, Golden
    except ImportError:
        raise MissingDependencyError(
            "PlanQuality with model requires deepeval.dataset.EvaluationDataset and Golden. "
            "Please ensure deepeval is up to date: pip install -U deepeval",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from None

    get_client_and_model()
    results: List[Dict[str, Any]] = []
    _, llm_model = get_client_and_model()

    # Run one golden at a time so the metric runs after each predict_fn and
    # metric.score is set when the iterator exits (evals_iterator runs the
    # metric when advancing; with one golden we get one iteration then exit).
    for _, row in dataset._df.iterrows():
        golden = Golden(input=row[input_column])
        eval_dataset = EvaluationDataset(goldens=[golden])
        metric = PlanQualityMetric(
            threshold=threshold,
            model=llm_model,
            include_reason=True,
            strict_mode=strict_mode,
        )
        for golden in eval_dataset.evals_iterator(metrics=[metric]):
            model.predict_fn(
                {
                    "input": golden.input,
                    "session_id": str(uuid.uuid4()),
                }
            )
        # After the loop, iterator ran the metric for this golden; score is set.
        score = metric.score
        reason = getattr(metric, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
