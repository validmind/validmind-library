from typing import Dict, Any
from langchain_core.messages import ToolMessage


def capture_tool_output_messages(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture all tool outputs and metadata from agent results.

    Args:
        agent_result: The result from the LangChain agent execution
    Returns:
        Dictionary containing tool outputs and metadata
    """
    messages = agent_result.get('messages', [])
    tool_outputs = []

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_outputs.append({
                'tool_name': 'unknown',  # ToolMessage doesn't directly contain tool name
                'content': message.content,
                'tool_call_id': getattr(message, 'tool_call_id', None)
            })

    return {
        'tool_outputs': tool_outputs,
        'total_messages': len(messages),
        'tool_message_count': len(tool_outputs)
    }
