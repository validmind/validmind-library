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
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for ToolCorrectness. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


@scorer()
@tags("llm", "ToolCorrectness", "deepeval", "agent_evaluation", "action_layer")
@tasks("llm")
def ToolCorrectness(
    dataset: VMDataset,
    threshold: float = 0.7,
    input_column: str = "input",
    expected_tools_column: str = "expected_tools",
    tools_called_column: str = "tools_called",
    agent_output_column: str = "agent_output",
    actual_output_column: str = "actual_output",
) -> List[Dict[str, Any]]:
    """Evaluate tool-use correctness for LLM agents using deepeval's ToolCorrectnessMetric.

    This metric assesses whether the agent called the expected tools in a task, and whether
    argument and response information matches the ground truth expectations.
    The metric compares the tools the agent actually called to the list of expected tools
    on a per-row basis.

    Args:
        dataset: VMDataset containing the agent input, expected tool calls, and actual tool calls.
        threshold: Minimum passing threshold (default: 0.7).
        input_column: Column containing the task input for evaluation.
        expected_tools_column: Column specifying the expected tools (ToolCall/str/dict or list).
        tools_called_column: Column holding the tools actually called by the agent.
            If missing, will be populated by parsing agent_output_column.
        agent_output_column: Column containing agent output with tool-calling trace (default: "agent_output").
        actual_output_column: Column specifying the ground-truth output string (optional).

    Returns:
        List of dicts (one per row) containing:
          - "score": Tool correctness score between 0 and 1.
          - "reason": ToolCorrectnessMetric's reason or explanation.

    Raises:
        ValueError: If required columns are missing from dataset.

    Example:
        results = ToolCorrectness(dataset=my_data)
        results[0]["score"]  # 1.0 if tools called correctly, else <1.0

    Risks & Limitations:
        - Works best if dataset includes high-quality tool call signals & references.
        - Comparison logic may be limited for atypically formatted tool call traces.
    """
    # Validate required columns exist in dataset
    missing_columns: List[str] = []
    if input_column not in dataset._df.columns:
        missing_columns.append(input_column)
    if expected_tools_column not in dataset._df.columns:
        missing_columns.append(expected_tools_column)

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset._df.columns.tolist()}"
        )

        # Import helper functions to avoid circular import
    from validmind.scorer.llm.deepeval import (
        _convert_to_tool_call_list,
        extract_tool_calls_from_agent_output,
    )

    _, model = get_client_and_model()

    metric = ToolCorrectnessMetric(
        threshold=threshold,
        model=model,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset._df.iterrows():
        input_value = row[input_column]
        expected_tools_value = row.get(expected_tools_column, [])

        # Extract tools called
        if tools_called_column in dataset._df.columns:
            tools_called_value = row.get(tools_called_column, [])
        else:
            agent_output = row.get(agent_output_column, {})
            tools_called_value = extract_tool_calls_from_agent_output(agent_output)

        expected_tools_list = _convert_to_tool_call_list(expected_tools_value)
        tools_called_list = _convert_to_tool_call_list(tools_called_value)

        actual_output_value = row.get(actual_output_column, "")

        test_case = LLMTestCase(
            input=input_value,
            expected_tools=expected_tools_list,
            tools_called=tools_called_list,
            actual_output=actual_output_value,
            _trace_dict=row.get(agent_output_column, {}),
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
