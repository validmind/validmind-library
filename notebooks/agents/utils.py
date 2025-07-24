from typing import Dict, Any
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
