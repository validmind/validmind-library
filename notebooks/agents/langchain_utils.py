from typing import Dict, List, Any
from langchain_core.messages import ToolMessage, AIMessage


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


def extract_tool_results_only(agent_result: Dict[str, Any]) -> List[str]:
    """
    Extract just the tool results in a simple format.
    
    Args:
        agent_result: The result from the LangChain agent execution
        
    Returns:
        List of tool result strings
    """
    messages = agent_result.get('messages', [])
    tool_results = []
    
    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results.append(message.content)
    
    return tool_results


def get_final_agent_response(agent_result: Dict[str, Any]) -> str:
    """
    Get the final agent response from the conversation.
    
    Args:
        agent_result: The result from the LangChain agent execution
        
    Returns:
        The final response content as a string
    """
    messages = agent_result.get('messages', [])
    
    # Look for the last AI message
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content
    
    return "No final response found"


def format_tool_outputs_for_display(captured_data: Dict[str, Any]) -> str:
    """
    Format tool outputs for readable display.
    
    Args:
        captured_data: Data from capture_tool_output_messages
        
    Returns:
        Formatted string for display
    """
    output = "Tool Execution Summary:\n"
    output += f"Total messages: {captured_data['total_messages']}\n"
    output += f"Tool messages: {captured_data['tool_message_count']}\n\n"
    
    for i, tool_output in enumerate(captured_data['tool_outputs'], 1):
        output += f"Tool {i}: {tool_output['tool_name']}\n"
        output += f"Output: {tool_output['content']}\n"
        output += "-" * 30 + "\n"
    
    return output
