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
    from deepeval.metrics import TaskCompletionMetric
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for TaskCompletion. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "TaskCompletion", "deepeval", "agentic")
@tasks("llm")
def TaskCompletion(
    dataset: VMDataset,
    model: Optional[VMModel] = None,
    threshold: float = 0.5,
    input_column: str = "input",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates agent task completion using deepeval's TaskCompletionMetric.

    This metric assesses whether the agent's output completes the requested task.

    When ``model`` is provided, the agent is run per row inside deepeval's evals_iterator
    so the metric receives trace data. Without ``model``, the dataset-only path uses
    pre-computed columns.

    Args:
        dataset: Dataset containing the agent input and final output
        model: Optional ValidMind model (agent) with predict_fn. When provided, the
            agent is run per row inside deepeval's evals_iterator so the metric
            receives trace data.
        threshold: Minimum passing threshold (default: 0.5)
        input_column: Column name for the task input (default: "input")
        actual_output_column: Column for the agent's final output (default: "actual_output")
        strict_mode: If True, enforces a binary score (0 for perfect, 1 otherwise)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """
    # Trace-based path: run agent inside evals_iterator so metric sees the trace
    if model is not None and getattr(model, "predict_fn", None) is not None:
        try:
            from deepeval.dataset import EvaluationDataset, Golden
        except ImportError:
            raise MissingDependencyError(
                "TaskCompletion with model requires deepeval.dataset.EvaluationDataset and Golden. "
                "Please ensure deepeval is up to date: pip install -U deepeval",
                required_dependencies=["deepeval"],
                extra="llm",
            ) from None

        _, llm_model = get_client_and_model()
        results: List[Dict[str, Any]] = []

        for _, row in dataset._df.iterrows():
            golden = Golden(input=row[input_column])
            eval_dataset = EvaluationDataset(goldens=[golden])
            metric = TaskCompletionMetric(
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
            score = metric.score
            reason = getattr(metric, "reason", "No reason provided")
            results.append({"score": score, "reason": reason})

        return results
