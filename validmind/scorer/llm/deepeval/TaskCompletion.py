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
    from deepeval.metrics import TaskCompletionMetric
    from deepeval.test_case import LLMTestCase, ToolCall
except ImportError as e:
    if "deepeval" in str(e):
        raise MissingDependencyError(
            "Missing required package `deepeval` for TaskCompletion. "
            "Please run `pip install validmind[llm]` to use LLM tests",
            required_dependencies=["deepeval"],
            extra="llm",
        ) from e

    raise e


def _extract_tool_responses(messages: List[Any]) -> Dict[str, str]:
    """Extract tool responses from messages."""
    tool_responses = {}

    for message in messages:
        # Handle both object and dictionary formats
        if isinstance(message, dict):
            # Dictionary format
            if (
                message.get("name")
                and message.get("content")
                and message.get("tool_call_id")
            ):
                tool_responses[message["tool_call_id"]] = message["content"]
        else:
            # Object format
            if hasattr(message, "name") and hasattr(message, "content"):
                if hasattr(message, "tool_call_id"):
                    tool_responses[message.tool_call_id] = message.content

    return tool_responses


def _extract_tool_calls_from_message(
    message: Any, tool_responses: Dict[str, str]
) -> List[ToolCall]:
    """Extract tool calls from a single message."""
    tool_calls = []

    # Handle both object and dictionary formats
    if isinstance(message, dict):
        # Dictionary format
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")

                if tool_name and tool_id:
                    # Get the response for this tool call
                    response = tool_responses.get(tool_id, "")

                    # Create ToolCall object
                    tool_call_obj = ToolCall(
                        name=tool_name, input_parameters=tool_args, output=response
                    )
                    tool_calls.append(tool_call_obj)
    else:
        # Object format
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                # Handle both dictionary and object formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")
                else:
                    # ToolCall object
                    tool_name = getattr(tool_call, "name", None)
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", None)

                if tool_name and tool_id:
                    # Get the response for this tool call
                    response = tool_responses.get(tool_id, "")

                    # Create ToolCall object
                    tool_call_obj = ToolCall(
                        name=tool_name, input_parameters=tool_args, output=response
                    )
                    tool_calls.append(tool_call_obj)

    return tool_calls


def extract_tool_calls_from_agent_output(
    agent_output: Dict[str, Any]
) -> List[ToolCall]:
    """
    Extract tool calls from the banking_agent_model_output column.

    Args:
        agent_output: The dictionary from banking_agent_model_output column

    Returns:
        List of ToolCall objects with name, args, and response
    """
    tool_calls = []

    if not isinstance(agent_output, dict) or "messages" not in agent_output:
        return tool_calls

    messages = agent_output["messages"]

    # First pass: collect tool responses
    tool_responses = _extract_tool_responses(messages)

    # Second pass: extract tool calls and match with responses
    for message in messages:
        message_tool_calls = _extract_tool_calls_from_message(message, tool_responses)
        tool_calls.extend(message_tool_calls)

    return tool_calls


# Create custom ValidMind tests for DeepEval metrics
@scorer()
@tags("llm", "TaskCompletion", "deepeval")
@tasks("llm")
def TaskCompletion(
    dataset: VMDataset,
    threshold: float = 0.5,
    input_column: str = "input",
    actual_output_column: str = "actual_output",
    agent_output_column: str = "banking_agent_model_output",
    tools_called_column: str = "tools_called",
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates agent task completion using deepeval's TaskCompletionMetric.

    This metric assesses whether the agent's output completes the requested task.

    Args:
        dataset: Dataset containing the agent input and final output
        threshold: Minimum passing threshold (default: 0.5)
        input_column: Column name for the task input (default: "input")
        actual_output_column: Column for the agent's final output (default: "actual_output")
        strict_mode: If True, enforces a binary score (0 for perfect, 1 otherwise)

    Returns:
        List[Dict[str, Any]] with keys "score" and "reason" for each row.

    Raises:
        ValueError: If required columns are missing
    """

    # Validate required columns exist in dataset
    missing_columns: List[str] = []
    for col in [input_column, actual_output_column]:
        if col not in dataset._df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset. "
            f"Available columns: {dataset.df.columns.tolist()}"
        )

    _, model = get_client_and_model()

    metric = TaskCompletionMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        strict_mode=strict_mode,
        verbose_mode=False,
    )

    results: List[Dict[str, Any]] = []
    for _, row in dataset._df.iterrows():
        input_value = row[input_column]
        actual_output_value = row[actual_output_column]
        if tools_called_column in dataset._df.columns:
            all_tool_calls = row[tools_called_column]
        else:
            agent_output = row.get(agent_output_column, {})
            all_tool_calls = extract_tool_calls_from_agent_output(agent_output)

        print(all_tool_calls)
        test_case = LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
            tools_called=all_tool_calls,
        )

        result = evaluate(test_cases=[test_case], metrics=[metric])
        metric_data = result.test_results[0].metrics_data[0]
        score = metric_data.score
        reason = getattr(metric_data, "reason", "No reason provided")
        results.append({"score": score, "reason": reason})

    return results
