# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from typing import Any, Dict, List

try:
    from deepeval.test_case import ToolCall
except ImportError:
    ToolCall = None  # type: ignore

from .AnswerRelevancy import AnswerRelevancy
from .ArgumentCorrectness import ArgumentCorrectness
from .PlanAdherence import PlanAdherence
from .PlanQuality import PlanQuality
from .ToolCorrectness import ToolCorrectness

__all__ = [
    "AnswerRelevancy",
    "ArgumentCorrectness",
    "PlanAdherence",
    "PlanQuality",
    "ToolCorrectness",
    "_extract_tool_responses",
    "_extract_tool_calls_from_message",
    "extract_tool_calls_from_agent_output",
    "_convert_to_tool_call_list",
]


def _extract_tool_responses(messages: List[Any]) -> Dict[str, str]:
    """Extract tool responses from the provided message list.

    Args:
        messages: List of message objects or dictionaries.

    Returns:
        Dictionary mapping tool_call_id to the tool's response content.
    """
    tool_responses = {}

    for message in messages:
        # Handle both object and dictionary formats
        if isinstance(message, dict):
            if (
                message.get("name")
                and message.get("content")
                and message.get("tool_call_id")
            ):
                tool_responses[message["tool_call_id"]] = message["content"]
        else:
            if hasattr(message, "name") and hasattr(message, "content"):
                if hasattr(message, "tool_call_id"):
                    tool_responses[message.tool_call_id] = message.content

    return tool_responses


def _extract_tool_calls_from_message(
    message: Any, tool_responses: Dict[str, str]
) -> List[ToolCall]:
    """Extract tool calls from a single message.

    Args:
        message: A message object or dict, possibly with tool_calls.
        tool_responses: Dict mapping tool_call_id to response content.

    Returns:
        List of ToolCall objects for all tool calls in message.
    """
    tool_calls = []

    if isinstance(message, dict):
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                if tool_name and tool_id:
                    response = tool_responses.get(tool_id, "")
                    tool_call_obj = ToolCall(
                        name=tool_name, input_parameters=tool_args, output=response
                    )
                    tool_calls.append(tool_call_obj)
    else:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")
                else:
                    tool_name = getattr(tool_call, "name", None)
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", None)
                if tool_name and tool_id:
                    response = tool_responses.get(tool_id, "")
                    tool_call_obj = ToolCall(
                        name=tool_name, input_parameters=tool_args, output=response
                    )
                    tool_calls.append(tool_call_obj)

    return tool_calls


def extract_tool_calls_from_agent_output(
    agent_output: Dict[str, Any]
) -> List[ToolCall]:
    """Extract ToolCall objects from an agent's output.

    Args:
        agent_output: The dictionary from the agent_output column,
            expected to contain a "messages" key.

    Returns:
        List of ToolCall objects with name, input_parameters, and output.
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


def _convert_to_tool_call_list(tools: Any) -> List[ToolCall]:
    """Convert a list of tool dicts/ToolCall/str into a list of ToolCall objects.

    Args:
        tools: List of tools in ToolCall, dict, or str format.
            If already a list of ToolCall objects, returns them as-is.

    Returns:
        List of ToolCall objects.
    """
    if ToolCall is None:
        raise ImportError(
            "deepeval.test_case.ToolCall is not available. "
            "Please install deepeval: pip install deepeval"
        )

    if not isinstance(tools, list):
        tools = []

    # If the input is already a list of ToolCall objects, return it directly
    if tools and all(isinstance(tool, ToolCall) for tool in tools):
        return tools

    tool_call_list = []
    for tool in tools:
        if isinstance(tool, ToolCall):
            tool_call_list.append(tool)
        elif isinstance(tool, dict):
            tool_call_list.append(
                ToolCall(
                    name=tool.get("name", ""),
                    input_parameters=tool.get("input_parameters", {}),
                    output=tool.get("output", ""),
                )
            )
        elif isinstance(tool, str):
            tool_call_list.append(ToolCall(name=tool))
    return tool_call_list
