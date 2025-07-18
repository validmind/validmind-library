from typing import Dict, List, Any, Optional
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage


def capture_tool_output_messages(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture and extract tool output messages from LangGraph agent results.

    Args:
        result: The result dictionary from a LangGraph agent execution

    Returns:
        Dictionary containing organized tool outputs and metadata
    """
    captured_data = {
        "tool_outputs": [],
        "tool_calls": [],
        "ai_responses": [],
        "human_inputs": [],
        "execution_summary": {},
        "message_flow": []
    }

    messages = result.get("messages", [])

    # Process each message in the conversation
    for i, message in enumerate(messages):
        message_info = {
            "index": i,
            "type": type(message).__name__,
            "content": getattr(message, 'content', ''),
            "timestamp": getattr(message, 'timestamp', None)
        }

        if isinstance(message, HumanMessage):
            captured_data["human_inputs"].append({
                "index": i,
                "content": message.content,
                "message_id": getattr(message, 'id', None)
            })
            message_info["category"] = "human_input"

        elif isinstance(message, AIMessage):
            # Capture AI responses
            ai_response = {
                "index": i,
                "content": message.content,
                "message_id": getattr(message, 'id', None)
            }

            # Check for tool calls in the AI message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_info = []
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        tool_call_info = {
                            "name": tool_call.get('name'),
                            "args": tool_call.get('args'),
                            "id": tool_call.get('id')
                        }
                    else:
                        # ToolCall object
                        tool_call_info = {
                            "name": getattr(tool_call, 'name', None),
                            "args": getattr(tool_call, 'args', {}),
                            "id": getattr(tool_call, 'id', None)
                        }
                    tool_calls_info.append(tool_call_info)
                    captured_data["tool_calls"].append(tool_call_info)

                ai_response["tool_calls"] = tool_calls_info
                message_info["category"] = "ai_with_tool_calls"
            else:
                message_info["category"] = "ai_response"

            captured_data["ai_responses"].append(ai_response)

        elif isinstance(message, ToolMessage):
            # Capture tool outputs
            tool_output = {
                "index": i,
                "tool_name": getattr(message, 'name', 'unknown'),
                "content": message.content,
                "tool_call_id": getattr(message, 'tool_call_id', None),
                "message_id": getattr(message, 'id', None)
            }
            captured_data["tool_outputs"].append(tool_output)
            message_info["category"] = "tool_output"
            message_info["tool_name"] = tool_output["tool_name"]

        captured_data["message_flow"].append(message_info)

    # Create execution summary
    captured_data["execution_summary"] = {
        "total_messages": len(messages),
        "tool_calls_count": len(captured_data["tool_calls"]),
        "tool_outputs_count": len(captured_data["tool_outputs"]),
        "ai_responses_count": len(captured_data["ai_responses"]),
        "human_inputs_count": len(captured_data["human_inputs"]),
        "tools_used": list(set([output["tool_name"] for output in captured_data["tool_outputs"]])),
        "conversation_complete": len(captured_data["tool_outputs"]) == len(captured_data["tool_calls"])
    }

    return captured_data


def extract_tool_results_only(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract only the tool results/outputs in a simplified format.

    Args:
        result: The result dictionary from a LangGraph agent execution

    Returns:
        List of dictionaries with tool name and output content
    """
    tool_results = []
    messages = result.get("messages", [])

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results.append({
                "tool_name": getattr(message, 'name', 'unknown'),
                "output": message.content,
                "tool_call_id": getattr(message, 'tool_call_id', None)
            })

    return tool_results


def get_final_agent_response(result: Dict[str, Any]) -> Optional[str]:
    """
    Get the final response from the agent (last AI message).

    Args:
        result: The result dictionary from a LangGraph agent execution

    Returns:
        The content of the final AI message, or None if not found
    """
    messages = result.get("messages", [])

    # Find the last AI message
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content

    return None


def format_tool_outputs_for_display(captured_data: Dict[str, Any]) -> str:
    """
    Format tool outputs in a readable string format.

    Args:
        captured_data: Result from capture_tool_output_messages()

    Returns:
        Formatted string representation of tool outputs
    """
    output_lines = []
    output_lines.append("🔧 TOOL OUTPUTS SUMMARY")
    output_lines.append("=" * 40)

    summary = captured_data["execution_summary"]
    output_lines.append(f"Total tools used: {len(summary['tools_used'])}")
    output_lines.append(f"Tools: {', '.join(summary['tools_used'])}")
    output_lines.append(f"Tool calls: {summary['tool_calls_count']}")
    output_lines.append(f"Tool outputs: {summary['tool_outputs_count']}")
    output_lines.append("")

    for i, output in enumerate(captured_data["tool_outputs"], 1):
        output_lines.append(f"{i}. {output['tool_name'].upper()}")
        output_lines.append(f"   Output: {output['content'][:100]}{'...' if len(output['content']) > 100 else ''}")
        output_lines.append("")

    return "\n".join(output_lines)


# Example usage functions
def demo_capture_usage(agent_result):
    """Demonstrate how to use the capture functions."""

    # Capture all tool outputs and metadata
    captured = capture_tool_output_messages(agent_result)

    # Get just the tool results
    tool_results = extract_tool_results_only(agent_result)

    # Get the final agent response
    final_response = get_final_agent_response(agent_result)

    # Format for display
    formatted_output = format_tool_outputs_for_display(captured)

    return {
        "full_capture": captured,
        "tool_results_only": tool_results,
        "final_response": final_response,
        "formatted_display": formatted_output
    }
