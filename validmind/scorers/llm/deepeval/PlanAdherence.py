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
    from deepeval.metrics import PlanAdherenceMetric
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for PlanAdherence. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


@scorer()
@tags(
    "llm", "PlanAdherence", "deepeval", "agent_evaluation", "reasoning_layer", "agentic"
)
@tasks("llm")
def PlanAdherence(
    dataset: VMDataset,
    model: Optional[VMModel] = None,
    threshold: float = 0.7,
    input_column: str = "input",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluates whether an agent follows its generated plan during execution, using deepeval's PlanAdherenceMetric.

    Plan adherence is critical in agentic reasoning: even the best plans become irrelevant if not executed faithfully.
    This scorer measures whether each agent output (per row) aligns with the agent's plan.

    If ``model`` is provided, each dataset row triggers a full agent run inside deepeval's evals_iterator so plan adherence is assessed from real traces. If ``model`` is not provided, the metric operates on pre-computed columns.

    Args:
        dataset (VMDataset): Dataset containing agent input, plan, and execution step columns.
        model (Optional[VMModel]): ValidMind agent model with a ``predict_fn``. Required for trace-based evaluation.
        threshold (float): Passing threshold for adherence score. Defaults to 0.7.
        input_column (str): Column name for the main agent prompt or task input. Defaults to "input".
        strict_mode (bool): If True, only returns scores 0 or 1 (binary adherence).

    Returns:
        List[Dict[str, Any]]: For each row, returns a dict with the final plan adherence "score" (float) and "reason" (str) from the metric.

    Raises:
        ValueError: If one or more required dataset columns are missing.
        MissingDependencyError: If required deepeval dependencies are not installed.

    Example:
        >>> results = PlanAdherence(dataset=my_ds, model=my_agent, threshold=0.8, input_column="user_input")
        >>> print(results[0]["score"], results[0]["reason"])
    """
    # Trace-based path: run agent inside evals_iterator so PlanAdherenceMetric sees the trace
    if model is None or not hasattr(model, "predict_fn") or model.predict_fn is None:
        raise ValueError(
            "PlanAdherence requires a `model` with a callable `predict_fn` for trace-based evaluation."
        )
    try:
        from deepeval.dataset import EvaluationDataset, Golden
    except ImportError:
        raise MissingDependencyError(
            "PlanAdherence with model requires deepeval.dataset.EvaluationDataset and Golden. "
            "Please ensure deepeval is up to date: pip install -U deepeval",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from None

    _, llm_model = get_client_and_model()
    results: List[Dict[str, Any]] = []

    # Run one golden at a time so the metric runs after each predict_fn and
    # metric.score is set when the iterator exits.
    for _, row in dataset._df.iterrows():
        golden = Golden(input=row[input_column])
        eval_dataset = EvaluationDataset(goldens=[golden])
        metric = PlanAdherenceMetric(
            threshold=threshold,
            model=llm_model,
            include_reason=True,
            strict_mode=strict_mode,
            verbose_mode=False,
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
