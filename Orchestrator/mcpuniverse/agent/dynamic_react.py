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
from mcpuniverse.agent.context_manager import ContextManager


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
        config: Dict | str = None,
        enable_compression: bool = True,
        model_context_limit: int = 200000,
    ):
        """
        Initialize the dynamic ReAct agent.

        Args:
            mcp_manager: MCP manager instance
            llm: Language model instance
            server_configs: Dictionary of all available server configurations
            config: Agent configuration
            enable_compression: Enable two-layer context compression
            model_context_limit: Model's context window size (in characters)
        """
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self.server_configs = server_configs
        self.loaded_servers = set()  # Tracks ALL loaded servers
        self.dynamically_loaded_servers = set()  # Tracks ONLY dynamically loaded servers
        self.trajectory = []  # Store tool call trajectory (for backward compatibility)
        self.reasoning_trace = []  # Store complete reasoning trace (thoughts + actions + observations)

        # Initialize context manager for compression
        self.enable_compression = enable_compression
        if enable_compression:
            self.context_manager = ContextManager(
                llm=llm,
                model_context_limit=model_context_limit,
            )
            self._logger.info("✓ Context compression enabled (two-layer strategy)")
        else:
            self.context_manager = None
            self._logger.info("Context compression disabled")

    def _add_history(self, history_type: str, message: str):
        """
        Override to capture reasoning trace in addition to history.
        """
        # Call parent to add to history
        super()._add_history(history_type, message)

        # Also save to reasoning trace for trajectory
        self.reasoning_trace.append({
            "type": history_type,
            "content": message
        })

    def _build_prompt(self, question: str):
        """
        Override to apply Layer 2 compression to history before building prompt.

        This ensures that when context_manager compresses the trajectory,
        those compressions are reflected in the prompt sent to the LLM.
        """
        # If Layer 2 compression has been applied, rebuild history from compressed trajectory
        if self.enable_compression and self.context_manager and self.context_manager.trajectory:
            # Sync parent's _history with compressed trajectory
            self._sync_history_from_trajectory()

        # Call parent to build prompt with potentially compressed history
        return super()._build_prompt(question)

    def _sync_history_from_trajectory(self):
        """
        Synchronize parent's _history with compressed trajectory from context_manager.

        This is called before building prompts to ensure Layer 2 compressions
        are reflected in the history sent to the LLM.
        """
        if not self.context_manager or not self.context_manager.trajectory:
            return

        # Rebuild _history from compressed trajectory
        # Clear all "result" entries and rebuild from trajectory
        new_history = []
        result_index = 0

        for item in self._history:
            # Keep non-result items as-is
            if not item.startswith("Result:"):
                new_history.append(item)
            else:
                # Replace with compressed result from trajectory
                if result_index < len(self.context_manager.trajectory):
                    compressed_result = self.context_manager.trajectory[result_index]["result"]
                    new_history.append(f"Result: {compressed_result}")
                    result_index += 1
                else:
                    # Fallback: keep original if trajectory is shorter
                    new_history.append(item)

        self._history = new_history

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

    async def _load_server_on_demand(self, server_name: str) -> tuple[bool, str]:
        """
        Load an MCP server dynamically if not already loaded.

        Args:
            server_name: Name of the server to load

        Returns:
            Tuple of (success: bool, error_reason: str)
        """
        # Already loaded
        if server_name in self.loaded_servers:
            return (True, "")

        # Server config not found
        if server_name not in self.server_configs:
            self._logger.warning(f"⚠ No configuration found for server: {server_name}")
            return (False, "server_not_in_configs")

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

            return (True, "")

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            self._logger.error(f"✗ Failed to load server {server_name}: {e}")

            # Categorize error type
            if "403" in error_msg or "Forbidden" in error_msg:
                return (False, f"http_403_forbidden: {error_msg[:100]}")
            elif "404" in error_msg or "Not Found" in error_msg:
                return (False, f"http_404_not_found: {error_msg[:100]}")
            elif "timeout" in error_msg.lower():
                return (False, f"connection_timeout: {error_msg[:100]}")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                return (False, f"connection_error: {error_msg[:100]}")
            else:
                return (False, f"{error_type}: {error_msg[:100]}")

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

        load_error_reason = None
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
                    success, error_reason = await self._load_server_on_demand(server_name)
                    server_loaded_dynamically = success
                    load_error_reason = error_reason if not success else None

                    if not success:
                        # Return error as tool result
                        from mcp.types import CallToolResult, TextContent
                        return CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=f"Error: Server '{server_name}' could not be loaded. Reason: {error_reason}"
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

        # Process result and log to trajectory
        end_time = time.time()
        if tool_call:
            # ✨ LAYER 1: Compress single tool result if needed
            # Extract actual text content from CallToolResult
            result_str = result.content[0].text if (result and result.content) else None
            original_result_str = result_str  # Save original for logging

            if self.enable_compression and result_str and self.context_manager:
                try:
                    self._logger.info(f"🔍 Checking if tool result needs compression ({len(result_str)} chars)...")
                    compressed_str = await self.context_manager.process_tool_result(
                        tool_name=tool_call.get("tool", "unknown"),
                        tool_args=tool_call.get("arguments", {}),
                        result=result_str,
                    )

                    # ✨ CRITICAL: Replace result object content with compressed version
                    # This ensures parent class uses compressed result in history
                    if compressed_str != result_str and result:
                        from mcp.types import CallToolResult, TextContent
                        result = CallToolResult(
                            content=[TextContent(type="text", text=compressed_str)]
                        )
                        result_str = compressed_str  # Update for trajectory
                        self._logger.info(
                            f"✂️  Layer 1 Compression: "
                            f"{len(original_result_str)} → {len(compressed_str)} chars "
                            f"({len(compressed_str)/len(original_result_str)*100:.1f}%)"
                        )

                except Exception as e:
                    self._logger.error(f"❌ Failed to compress tool result: {e}")
                    # Keep original result on error

            # Build trajectory entry
            trajectory_entry = {
                "type": "tool_call",
                "thought": tool_call.get("thought"),  # Capture LLM's reasoning
                "server": tool_call.get("server"),
                "tool": tool_call.get("tool"),
                "arguments": tool_call.get("arguments"),
                "dynamically_loaded": server_loaded_dynamically,
                "duration_seconds": round(end_time - start_time, 3),
                "result_preview": result_str[:200] if result_str else None,
                "result": result_str,  # Compressed result from Layer 1
                "original_size": len(original_result_str) if original_result_str else 0,
                "compressed": result_str != original_result_str if original_result_str else False,
            }

            # Add server load failure reason if applicable
            if load_error_reason:
                trajectory_entry["load_error"] = load_error_reason
                trajectory_entry["status"] = "server_load_failed"
            # Add error info if tool call failed
            elif error:
                trajectory_entry["status"] = "error"
                trajectory_entry["error"] = error
            else:
                trajectory_entry["status"] = "success"

            # Add to local trajectory (for backward compatibility)
            self.trajectory.append(trajectory_entry)

            # ✨ LAYER 2: Add to context manager and trigger global compression if needed
            if self.enable_compression and self.context_manager:
                try:
                    await self.context_manager.add_turn(
                        thought=trajectory_entry["thought"],
                        action={
                            "server": trajectory_entry["server"],
                            "tool": trajectory_entry["tool"],
                            "arguments": trajectory_entry["arguments"],
                        },
                        result=result_str or "",
                    )
                except Exception as e:
                    self._logger.error(f"Failed to add turn to context manager: {e}")

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
