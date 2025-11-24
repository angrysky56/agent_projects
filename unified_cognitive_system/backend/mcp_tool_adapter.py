"""
MCP Tool Adapter

Converts MCP tool schemas to OpenAI-compatible function calling format.
Enables LLMs to discover and use MCP tools through function calling.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def mcp_tool_to_openai_function(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an MCP tool schema to OpenAI function calling format.

    Args:
        mcp_tool: MCP tool definition with 'name', 'description', and 'inputSchema'

    Returns:
        OpenAI-compatible function definition

    Example:
        MCP format:
        {
            "name": "read_file",
            "description": "Read file contents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        }

        Converts to OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }
        }
    """
    try:
        # Extract MCP tool components
        name = mcp_tool.get("name", "")
        description = mcp_tool.get("description", "")
        input_schema = mcp_tool.get("inputSchema", {})

        if not name:
            logger.warning(f"MCP tool missing name: {mcp_tool}")
            return None

        # Build OpenAI function format
        openai_function = {"type": "function", "function": {"name": name, "description": description or f"Execute {name} tool", "parameters": input_schema if input_schema else {"type": "object", "properties": {}, "required": []}}}

        return openai_function

    except Exception as e:
        logger.error(f"Error converting MCP tool to OpenAI format: {e}", exc_info=True)
        return None


async def get_available_tools_for_llm(mcp_client) -> List[Dict[str, Any]]:
    """
    Fetch all available MCP tools and convert them to OpenAI function format.

    Args:
        mcp_client: MCPClient instance

    Returns:
        List of OpenAI-compatible function definitions
    """
    try:
        if not mcp_client or not mcp_client.sessions:
            logger.info("No MCP client or sessions available for tool discovery")
            return []

        # Fetch all MCP tools
        mcp_tools = await mcp_client.list_tools()

        if not mcp_tools:
            logger.info("No MCP tools available")
            return []

        # Convert each tool to OpenAI format
        openai_functions = []
        for mcp_tool in mcp_tools:
            openai_func = mcp_tool_to_openai_function(mcp_tool)
            if openai_func:
                openai_functions.append(openai_func)

        logger.info(f"Converted {len(openai_functions)} MCP tools to OpenAI function format")
        return openai_functions

    except Exception as e:
        logger.error(f"Error fetching MCP tools for LLM: {e}", exc_info=True)
        return []


def format_tool_call_for_mcp(tool_call: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Extract tool name and arguments from LLM tool call response.

    Args:
        tool_call: Tool call from LLM response

    Returns:
        Tuple of (tool_name, arguments)

    Example LLM tool_call format:
        {
            "function": {
                "name": "read_file",
                "arguments": {"path": "/etc/os-release", "offset": 0, "length": 5}
            }
        }
    """
    try:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        arguments = function.get("arguments", {})

        # Arguments might be a JSON string, try to parse
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse arguments as JSON: {arguments}")
                arguments = {}

        return tool_name, arguments

    except Exception as e:
        logger.error(f"Error formatting tool call for MCP: {e}", exc_info=True)
        return "", {}


def format_tool_result_for_llm(tool_name: str, result: Any) -> Dict[str, Any]:
    """
    Format MCP tool execution result for LLM consumption.

    Args:
        tool_name: Name of the tool that was executed
        result: Result from MCP tool execution

    Returns:
        Message dict with role="tool" for LLM
    """
    try:
        # Extract content from MCP result
        # MCP results have structure: {"content": [...], "isError": bool}
        content_str = ""

        if isinstance(result, dict):
            # Handle MCP SDK CallToolResult format
            content_list = result.get("content", [])
            if content_list and isinstance(content_list, list):
                # Extract text from content items
                text_parts = []
                for item in content_list:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content_str = "\n".join(text_parts)
            else:
                # Fallback: stringify the result
                import json

                content_str = json.dumps(result, indent=2)
        else:
            content_str = str(result)

        # Return as tool message for LLM
        return {"role": "tool", "content": content_str, "tool_name": tool_name}

    except Exception as e:
        logger.error(f"Error formatting tool result for LLM: {e}", exc_info=True)
        return {"role": "tool", "content": f"Error executing tool: {str(e)}", "tool_name": tool_name}
