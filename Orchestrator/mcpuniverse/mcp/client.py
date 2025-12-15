"""
This module provides a client implementation for interacting with MCP (Model Control Protocol) servers.

It includes the MCPClient class, which offers methods to connect to MCP servers using either
stdio or SSE transport, list available tools, and execute tools on the server.
"""
# pylint: disable=broad-exception-caught
import asyncio
import os
import shutil
from datetime import timedelta
from contextlib import AsyncExitStack
from typing import Any, Optional, Union, List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    streamablehttp_client = None
from mcpuniverse.common.misc import AutodocABCMeta
from mcpuniverse.mcp.config import ServerConfig
from mcpuniverse.common.logger import get_logger
from mcpuniverse.callbacks.base import (
    BaseCallback,
    CallbackMessage,
    MessageType,
    Status,
    Event,
    send_message
)

load_dotenv()


class MCPClient(metaclass=AutodocABCMeta):
    """
    A client for interacting with MCP (Model Control Protocol) servers.

    This class provides methods to connect to MCP servers using either stdio or SSE transport,
    list available tools, and execute tools.
    """

    def __init__(self, name: str):
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._logger = get_logger(self.__class__.__name__)
        self._name = name
        self._project_id = ""
        # Stdio context
        self._stdio_context: Union[Any, None] = None
        # Server parameters
        self._server_params = None

    async def connect_to_stdio_server(self, config: ServerConfig, timeout: int = 20):
        """
        Initializes a connection to an MCP server using stdio transport.

        Args:
            config (ServerConfig): Configuration object containing server settings.
            timeout (int, optional): Connection timeout in seconds. Defaults to 20.

        Raises:
            ValueError: If the command in the config is invalid.
            Exception: If the connection fails.

        Note:
            This method sets up the connection and initializes the client session.
        """
        command = (
            shutil.which(config.stdio.command)
            if config.stdio.command in ["npx", "docker", "python", "python3"]
            else config.stdio.command
        )
        if command is None or command == "":
            raise ValueError("The command must be a valid string")

        envs = dict(os.environ)
        envs.update(config.env)
        server_params = StdioServerParameters(
            command=command,
            args=config.stdio.args,
            env=envs
        )
        try:
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout))
            )
            await session.initialize()
            self._session = session
            self._server_params = {
                "command": server_params.command,
                "args": server_params.args,
                "env": envs
            }
        except Exception as e:
            self._logger.error("Failed to initialize client %s: %s", self._name, str(e))
            await self.cleanup()
            raise e

    async def connect_to_sse_server(
        self,
        server_url: str,
        timeout: int = 20,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Any] = None
    ):
        """
        Connects to an MCP server using SSE (Server-Sent Events) transport.

        Args:
            server_url (str): The URL of the MCP server.
            timeout (int, optional): Connection timeout in seconds. Defaults to 20.
            headers (dict, optional): Custom headers to include in the request.
            auth (Any, optional): OAuth authentication provider (e.g., from create_smithery_auth).

        Raises:
            Exception: If the connection fails.

        Note:
            This method sets up the SSE connection and initializes the client session.
        """
        try:
            use_streamable_http = streamablehttp_client is not None and server_url.startswith(("http://", "https://"))
            if headers or auth:
                use_streamable_http = use_streamable_http or streamablehttp_client is not None

            if use_streamable_http and streamablehttp_client:
                log_msg = "Connecting to %s using streamablehttp_client"
                if headers:
                    log_msg = "Connecting to %s with custom headers using streamablehttp_client"
                if auth:
                    log_msg = "Connecting to %s with OAuth using streamablehttp_client"
                self._logger.info(log_msg, server_url)
                transport = await self._exit_stack.enter_async_context(
                    streamablehttp_client(server_url, headers=headers, auth=auth)
                )
                read, write, _ = transport
            else:
                # Fall back to regular SSE client
                self._logger.info("Connecting to %s using sse_client", server_url)
                sse_transport = await self._exit_stack.enter_async_context(sse_client(url=server_url, headers=headers))
                read, write = sse_transport

            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout))
            )
            await session.initialize()
            self._session = session
            self._server_params = {"type": "url", "url": server_url}
        except Exception as e:
            self._logger.error("Failed to initialize client %s: %s", self._name, str(e))
            await self.cleanup()
            raise e

    async def list_tools(self) -> list[Any]:
        """
        Retrieves a list of available tools from the connected MCP server.

        Returns:
            list[Any]: A list of available tools.

        Raises:
            RuntimeError: If the client is not initialized.
        """
        if not self._session:
            raise RuntimeError(f"Client {self._name} not initialized")

        tools_response = await self._session.list_tools()
        tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(tool)
        return tools

    async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            retries: int = 5,
            delay: float = 1.0,
            callbacks: BaseCallback | List[BaseCallback] = None,
    ) -> Any:
        """
        Executes a tool on the connected MCP server with a retry mechanism.

        Args:
            tool_name (str): The name of the tool to execute.
            arguments (dict[str, Any]): A dictionary of arguments for the tool.
            retries (int, optional): Number of retry attempts. Defaults to 5.
            delay (float, optional): Delay between retries in seconds. Defaults to 1.0.
            callbacks (BaseCallback | List[BaseCallback], optional):
                Callbacks for recording MCP call status and responses

        Returns:
            Any: The result of the tool execution.

        Raises:
            RuntimeError: If the client is not initialized or if all retry attempts fail.
        """
        if not self._session:
            raise RuntimeError(f"Client {self._name} not initialized")
        

        send_message(callbacks, message=CallbackMessage(
            source=self.id, type=MessageType.EVENT, data=Event.BEFORE_CALL,
            metadata={"method": "execute_tool"}, project_id=self._project_id))
        send_message(callbacks, message=CallbackMessage(
            source=self.id, type=MessageType.STATUS, data=Status.RUNNING,
            project_id=self._project_id))
        print(1)


        attempt = 0
        while attempt < retries:
            try:
                self._logger.info("Executing %s...", tool_name)
                print(2)
                result = await self._session.call_tool(tool_name, arguments)
                print(3)
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.RESPONSE,
                    data=result.model_dump(mode="json") if isinstance(result, BaseModel) else result,
                    project_id=self._project_id))
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.EVENT, data=Event.AFTER_CALL,
                    metadata={"method": "execute_tool"}, project_id=self._project_id))
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.STATUS, data=Status.SUCCEEDED,
                    project_id=self._project_id))
                return result

            except asyncio.CancelledError as e:
                # Handle task cancellation (e.g., from server errors during tool execution)
                self._logger.error("Tool execution cancelled: %s", str(e))
                # Create error result instead of crashing
                from mcp.types import CallToolResult, TextContent
                error_result = CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: Tool execution was cancelled (likely due to server timeout or error). Tool: {tool_name}"
                    )],
                    isError=True
                )
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.ERROR, data="Task cancelled",
                    project_id=self._project_id))
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.STATUS, data=Status.FAILED,
                    project_id=self._project_id))
                return error_result
            except Exception as e:
                attempt += 1
                self._logger.warning(
                    "Failed to execute tool: %s. Attempt %d of %d", str(e), attempt, retries
                )
                if attempt < retries:
                    self._logger.info("Retrying in %f seconds...", delay)
                    await asyncio.sleep(delay)
                else:
                    self._logger.error("Max retries reached")
                    send_message(callbacks, message=CallbackMessage(
                        source=self.id, type=MessageType.ERROR, data=str(e),
                        project_id=self._project_id))
                    send_message(callbacks, message=CallbackMessage(
                        source=self.id, type=MessageType.EVENT, data=Event.AFTER_CALL,
                        metadata={"method": "execute_tool"}, project_id=self._project_id))
                    send_message(callbacks, message=CallbackMessage(
                        source=self.id, type=MessageType.STATUS, data=Status.FAILED,
                        project_id=self._project_id))
                    raise e

    async def cleanup(self):
        """
        Cleans up client resources and closes the session.

        This method should be called when the client is no longer needed to ensure
        proper resource management and connection closure.
        """
        async with self._cleanup_lock:
            try:
                await self._exit_stack.aclose()
                self._session = None
                self._stdio_context = None
            except asyncio.CancelledError as e:
                # Suppress CancelledError during cleanup to prevent it from propagating
                self._logger.error("Cleanup cancelled for client %s: %s", self._name, str(e))
            except Exception as e:
                self._logger.error("Error during cleanup of client %s: %s", self._name, str(e))

    @property
    def project_id(self) -> str:
        """Return the ID of the project using this client."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: str):
        """Set the ID of the project using this client."""
        self._project_id = value

    @property
    def id(self):
        """Return the ID of this client."""
        if self._project_id:
            return f"{self._project_id}:mcp:{self._name}"
        return f"mcp:{self._name}"

    def get_mcp_config(self) -> Dict[str, Any]:
        """Return the MCP configuration for this client."""
        return self._server_params
