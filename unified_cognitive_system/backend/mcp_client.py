"""
MCP Client Integration

Provides a client interface for interacting with MCP servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for interacting with MCP servers.
    """

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self._server_params: Dict[str, StdioServerParameters] = {}

    async def connect(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Connect to an MCP server via stdio.

        Args:
            name: Server name
            command: Command to run server
            args: Arguments for command
            env: Environment variables
        """
        try:
            server_params = StdioServerParameters(command=command, args=args, env=env)
            self._server_params[name] = server_params

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport

            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self.sessions[name] = session

            logger.info(f"Connected to MCP server '{name}': {command} {' '.join(args)}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            # Don't raise, just log error so other servers can still connect

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from all connected servers."""
        all_tools = []

        for name, session in self.sessions.items():
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    tool_dump = tool.model_dump()
                    tool_dump["server_name"] = name
                    all_tools.append(tool_dump)
            except Exception as e:
                logger.error(f"Error listing tools from server '{name}': {e}")

        return all_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any], server_name: Optional[str] = None) -> Any:
        """
        Call a tool.

        Args:
            name: Tool name
            arguments: Tool arguments
            server_name: Optional specific server to call. If None, searches all servers.
        """
        if not self.sessions:
            raise RuntimeError("No MCP servers connected")

        # If server specified, use it
        if server_name and server_name in self.sessions:
            try:
                return await self.sessions[server_name].call_tool(name, arguments)
            except Exception as e:
                logger.error(f"Error calling tool {name} on server {server_name}: {e}")
                raise

        # Otherwise search for tool in all servers
        # Note: This is inefficient for frequent calls, but fine for now.
        # Ideally we'd cache tool->server mapping.
        for s_name, session in self.sessions.items():
            try:
                # We can't easily check if tool exists without listing,
                # so we try to call it and handle specific error if possible,
                # or we just rely on the user/system providing the correct server_name.
                # For now, let's try the first session if no server_name provided,
                # OR better, iterate and try. But 'call_tool' might have side effects.

                # BETTER STRATEGY: List tools to find the server if not provided
                tools_result = await session.list_tools()
                if any(t.name == name for t in tools_result.tools):
                    return await session.call_tool(name, arguments)
            except Exception:
                continue

        raise RuntimeError(f"Tool '{name}' not found on any connected server")

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from all connected servers."""
        all_resources = []

        for name, session in self.sessions.items():
            try:
                result = await session.list_resources()
                for resource in result.resources:
                    res_dump = resource.model_dump()
                    res_dump["server_name"] = name
                    all_resources.append(res_dump)
            except Exception as e:
                logger.error(f"Error listing resources from server '{name}': {e}")

        return all_resources

    async def read_resource(self, uri: str) -> Any:
        """Read a resource."""
        if not self.sessions:
            raise RuntimeError("No MCP servers connected")

        # Search all servers for the resource
        for name, session in self.sessions.items():
            try:
                # Ideally we check if resource exists first, but read_resource might throw if not found
                # or we can list resources first. For now, try reading.
                # Note: This might be problematic if multiple servers serve the same URI pattern.
                # But URIs should be unique.
                return await session.read_resource(uri)
            except Exception:
                continue

        raise RuntimeError(f"Resource '{uri}' not found on any connected server")

    async def shutdown(self):
        """Shutdown the client."""
        await self.exit_stack.aclose()
        self.sessions.clear()
        logger.info("MCP client shutdown")


# Global instance
_mcp_client: Optional[MCPClient] = None


async def get_mcp_client() -> MCPClient:
    """Get or create global MCP client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


async def initialize_mcp():
    """Initialize MCP by loading config and connecting to servers."""
    import json
    import os

    client = await get_mcp_client()

    # Load config
    config_path = "mcp_servers_config.json"
    if not os.path.exists(config_path):
        logger.warning(f"MCP config not found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})

        for name, server_config in servers.items():
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env")

            if command:
                await client.connect(name, command, args, env)

    except Exception as e:
        logger.error(f"Error initializing MCP from config: {e}")


async def shutdown_mcp():
    """Shutdown global MCP client."""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.shutdown()
        _mcp_client = None
