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
@tags("llm", "ArgumentCorrectness", "deepeval", "agent_evaluation", "action_layer")
@tasks("llm")
def ArgumentCorrectness(
    dataset: VMDataset,
    threshold: float = 0.7,
    input_column: str = "input",
    tools_called_column: str = "tools_called",
    agent_output_column: str = "agent_output",
    actual_output_column: str = "actual_output",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates agent argument correctness using deepeval's ArgumentCorrectnessMetric.

    This metric evaluates whether your agent generates correct arguments for each tool
    call. Selecting the right tool with wrong arguments is as problematic as selecting
    the wrong tool entirely.

    Unlike ToolCorrectnessMetric, this metric is fully LLM-based and referenceless—it
    evaluates argument correctness based on the input context rather than comparing
    against expected values.

    Args:
        dataset: Dataset containing the agent input and tool calls
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
    # Validate required columns exist in dataset
    missing_columns: List[str] = []
    if input_column not in dataset._df.columns:
        missing_columns.append(input_column)

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset._df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    metric = ArgumentCorrectnessMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )

    # Import helper functions to avoid circular import
    from validmind.scorers.llm.deepeval import (
        _convert_to_tool_call_list,
        extract_tool_calls_from_agent_output,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset._df.iterrows():
        input_value = row[input_column]

        # Extract tools called
        if tools_called_column in dataset._df.columns:
            tools_called_value = row.get(tools_called_column, [])
        else:
            agent_output = row.get(agent_output_column, {})
            tools_called_value = extract_tool_calls_from_agent_output(agent_output)
        tools_called_list = _convert_to_tool_call_list(tools_called_value)

        actual_output_value = row.get(actual_output_column, "")

        test_case = LLMTestCase(
            input=input_value,
            tools_called=tools_called_list,
            actual_output=actual_output_value,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
