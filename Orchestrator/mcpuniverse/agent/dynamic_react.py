#!/usr/bin/env python3
"""
Dynamic ReAct Agent - Automatically loads MCP servers on-demand

This wrapper around the ReAct agent intercepts tool calls and automatically
loads the required MCP servers when the agent discovers new tools via search.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List, Union

from mcpuniverse.agent.react import ReAct
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.tracer import Tracer
from mcpuniverse.callbacks.base import BaseCallback


class DynamicReActAgent(ReAct):
    """
    ReAct agent with dynamic server loading capability.

    This agent automatically loads MCP servers when it tries to use tools
    from servers that haven't been initialized yet.
    """

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        server_configs: Dict[str, Any],
        config: Dict | str = None
    ):
        """
        Initialize the dynamic ReAct agent.

        Args:
            mcp_manager: MCP manager instance
            llm: Language model instance
            server_configs: Dictionary of all available server configurations
            config: Agent configuration
        """
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self.server_configs = server_configs
        self.loaded_servers = set()  # Tracks ALL loaded servers
        self.dynamically_loaded_servers = set()  # Tracks ONLY dynamically loaded servers
        self.trajectory = []  # Store execution trajectory

    async def initialize(self, mcp_servers: List[Dict[str, str]] = None):
        """
        Initialize the agent with initial servers.
        Overrides parent to track which servers are pre-loaded.
        """
        # Call parent initialization
        await super().initialize(mcp_servers=mcp_servers)

        # Track initially loaded servers in loaded_servers (but NOT in dynamically_loaded_servers)
        if mcp_servers:
            for server in mcp_servers:
                server_name = server.get("name")
                if server_name:
                    self.loaded_servers.add(server_name)  # Track as loaded
                    self._logger.info(f"✓ Pre-loaded server: {server_name}")

    async def _load_server_on_demand(self, server_name: str) -> bool:
        """
        Load an MCP server dynamically if not already loaded.

        Args:
            server_name: Name of the server to load

        Returns:
            True if server was loaded successfully, False otherwise
        """
        # Already loaded
        if server_name in self.loaded_servers:
            return True

        # Server config not found
        if server_name not in self.server_configs:
            self._logger.warning(f"⚠ No configuration found for server: {server_name}")
            return False

        try:
            self._logger.info(f"🔄 Dynamically loading server: {server_name}")

            # Add server config to MCP manager
            config = self.server_configs[server_name]
            self._mcp_manager.add_server_config(server_name, config)

            # Set params to replace template variables (e.g., {{SMITHERY_API_KEY}})
            import os
            params = dict(os.environ)  # Use environment variables for template replacement
            self._mcp_manager.set_params(server_name, params)

            # Build client
            client = await self._mcp_manager.build_client(server_name, transport="streamable_http")
            self._mcp_clients[server_name] = client

            # Load tools
            tools = await client.list_tools()
            self._tools[server_name] = tools

            self.loaded_servers.add(server_name)
            self.dynamically_loaded_servers.add(server_name)  # Track as dynamically loaded
            self._logger.info(f"✅ Loaded {len(tools)} tools from {server_name}")

            return True

        except Exception as e:
            self._logger.error(f"✗ Failed to load server {server_name}: {e}")
            return False

    async def call_tool(
        self,
        llm_response: Union[str, Dict],
        tracer: Tracer = None,
        callbacks: BaseCallback | List[BaseCallback] = None,
    ):
        """
        Intercept tool calls and dynamically load servers if needed.

        This method wraps the parent's call_tool method to automatically
        load servers that the agent discovers via search_tools.
        """
        import time
        start_time = time.time()

        # Parse the LLM response to get server name
        tool_call = None
        server_loaded_dynamically = False

        try:
            if isinstance(llm_response, str):
                _response = llm_response.strip().strip('`').strip()
                if _response.startswith("json"):
                    _response = _response[4:].strip()
                tool_call = json.loads(_response)
            else:
                tool_call = llm_response

            # Check if we need to load the server
            if "server" in tool_call:
                server_name = tool_call["server"]

                # Try to load server if not already loaded
                if server_name not in self.loaded_servers:
                    self._logger.info(f"📡 Server {server_name} not loaded, attempting dynamic load...")
                    success = await self._load_server_on_demand(server_name)
                    server_loaded_dynamically = success

                    if not success:
                        # Return error as tool result
                        from mcp.types import CallToolResult, TextContent
                        return CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=f"Error: Server '{server_name}' could not be loaded. Make sure it's in the server configs."
                            )]
                        )

        except json.JSONDecodeError as e:
            self._logger.warning(f"Failed to parse tool call: {e}")
        except Exception as e:
            self._logger.warning(f"Error in dynamic server loading: {e}")

        # Call parent's call_tool method and capture success/failure
        result = None
        error = None
        try:
            result = await super().call_tool(llm_response, tracer, callbacks)
        except Exception as e:
            error = str(e)
            self._logger.error(f"Tool call failed: {e}")
            # Re-raise so the agent sees the error
            raise

        # Log to trajectory
        end_time = time.time()
        if tool_call:
            trajectory_entry = {
                "type": "tool_call",
                "thought": tool_call.get("thought"),  # Capture LLM's reasoning
                "server": tool_call.get("server"),
                "tool": tool_call.get("tool"),
                "arguments": tool_call.get("arguments"),
                "dynamically_loaded": server_loaded_dynamically,
                "duration_seconds": round(end_time - start_time, 3),
                "result_preview": str(result)[:200] if result else None,
            }

            # Add error info if tool call failed
            if error:
                trajectory_entry["status"] = "error"
                trajectory_entry["error"] = error
            else:
                trajectory_entry["status"] = "success"

            self.trajectory.append(trajectory_entry)

        return result

    async def cleanup(self):
        """Properly cleanup all MCP clients to avoid asyncio errors."""
        self._logger.info("Cleaning up MCP clients...")

        # Close all dynamically loaded clients
        for server_name in list(self.loaded_servers):
            if server_name in self._mcp_clients:
                try:
                    client = self._mcp_clients[server_name]
                    if hasattr(client, 'close'):
                        await client.close()
                    elif hasattr(client, '__aexit__'):
                        await client.__aexit__(None, None, None)
                except Exception as e:
                    self._logger.warning(f"Error closing client {server_name}: {e}")

        # Close the parent agent's clients
        try:
            await super().cleanup()
        except AttributeError:
            # Parent class might not have cleanup method
            pass

        self._logger.info("✓ Cleanup complete")
