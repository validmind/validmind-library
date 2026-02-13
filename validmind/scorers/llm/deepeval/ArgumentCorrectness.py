# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

from validmind import tags, tasks
from validmind.ai.utils import get_client_and_model
from validmind.errors import MissingDependencyError
from validmind.tests.decorator import scorer
from validmind.vm_models.dataset import VMDataset

try:
    from deepeval import evaluate
    from deepeval.metrics import ArgumentCorrectnessMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for ArgumentCorrectness. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


@scorer()
@tags(
    "llm",
    "ArgumentCorrectness",
    "deepeval",
    "agent_evaluation",
    "action_layer",
    "agentic",
)
@tasks("llm")
def ArgumentCorrectness(
    dataset: VMDataset,
    threshold: float = 0.7,
    input_column: str = "input",
    actual_tools_called_column: str = "tools_called",
) -> List[Dict[str, Any]]:
    """Evaluates agent argument correctness using deepeval's ArgumentCorrectnessMetric.

    This metric evaluates whether your agent generates correct arguments for each tool
    call. Selecting the right tool with wrong arguments is as problematic as selecting
    the wrong tool entirely.

    Unlike ToolCorrectnessMetric, this metric is fully LLM-based and referenceless—it
    evaluates argument correctness based on the input context rather than comparing
    against expected values.

    When ``model`` is provided, the agent is run per row inside deepeval's evals_iterator
    so the metric receives trace data. Without ``model``, the dataset-only path uses
    pre-computed columns.

    Args:
        dataset: Dataset containing the agent input and tool calls
        model: Optional ValidMind model (agent) with predict_fn. When provided, the
            agent is run per row inside deepeval's evals_iterator so the metric
            receives trace data.
        threshold: Minimum passing threshold (default: 0.7)
        input_column: Column name for the task input (default: "input")
        tools_called_column: Column name for tools called (default: "tools_called")
        agent_output_column: Column name for agent output containing tool calls (default: "agent_output")
        strict_mode: If True, enforces a binary score (0 or 1)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """
    from validmind.scorers.llm.deepeval import _convert_to_tool_call_list

    missing_columns: List[str] = []
    if input_column not in dataset._df.columns:
        missing_columns.append(input_column)
    if actual_tools_called_column not in dataset._df.columns:
        missing_columns.append(actual_tools_called_column)
    if missing_columns:
        raise ValueError(
            f"ToolCorrectness with model requires columns {missing_columns}. "
            f"Available columns: {dataset._df.columns.tolist()}"
        )

    _, llm_model = get_client_and_model()
    results: List[Dict[str, Any]] = []

    for _, row in dataset._df.iterrows():
        actual_tools_value = row.get(actual_tools_called_column, [])
        actual_tools_list = _convert_to_tool_call_list(actual_tools_value)

        metric = ArgumentCorrectnessMetric(
            threshold=threshold,
            model=llm_model,
        )

        test_case = LLMTestCase(
            input=row[input_column],
            tools_called=actual_tools_list,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        results.append({"score": score, "reason": metric_data.reason})
    return results
