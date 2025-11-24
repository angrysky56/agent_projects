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
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._server_params: Optional[StdioServerParameters] = None

    async def connect(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Connect to an MCP server via stdio.

        Args:
            command: Command to run server
            args: Arguments for command
            env: Environment variables
        """
        try:
            self._server_params = StdioServerParameters(command=command, args=args, env=env)

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(self._server_params))
            self.read, self.write = stdio_transport

            self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
            await self.session.initialize()

            logger.info(f"Connected to MCP server: {command} {' '.join(args)}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.session:
            return []

        try:
            result = await self.session.list_tools()
            return [tool.model_dump() for tool in result.tools]
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool."""
        if not self.session:
            raise RuntimeError("MCP client not connected")

        try:
            result = await self.session.call_tool(name, arguments)
            return result
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            raise

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources."""
        if not self.session:
            return []

        try:
            result = await self.session.list_resources()
            return [resource.model_dump() for resource in result.resources]
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            return []

    async def read_resource(self, uri: str) -> Any:
        """Read a resource."""
        if not self.session:
            raise RuntimeError("MCP client not connected")

        try:
            result = await self.session.read_resource(uri)
            return result
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise

    async def shutdown(self):
        """Shutdown the client."""
        await self.exit_stack.aclose()
        self.session = None
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
    """Initialize MCP (placeholder for auto-connection logic)."""
    # In a real scenario, we might auto-connect to configured servers here
    pass


async def shutdown_mcp():
    """Shutdown global MCP client."""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.shutdown()
        _mcp_client = None
