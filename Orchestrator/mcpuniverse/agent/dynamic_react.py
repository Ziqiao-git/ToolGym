#!/usr/bin/env python3
"""
Dynamic ReAct Agent - Automatically loads MCP servers on-demand

This wrapper around the ReAct agent intercepts tool calls and automatically
loads the required MCP servers when the agent discovers new tools via search.
"""
from __future__ import annotations

import asyncio
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
        enable_compression: bool = False,
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
        self.discovered_tools = {}  # Maps server -> set of tool names actually discovered/used
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
        Override to:
        1. Apply Layer 2 compression to history before building prompt
        2. Only show meta-mcp + specifically discovered tools (not all tools from servers)

        This encourages the agent to use search_tools for new capabilities while
        remembering only the specific tools it has discovered and used.
        """
        # If Layer 2 compression has been applied, rebuild history from compressed trajectory
        if self.enable_compression and self.context_manager and self.context_manager.trajectory:
            # Sync parent's _history with compressed trajectory
            self._sync_history_from_trajectory()

        # Filter tools to show: meta-mcp + ONLY the specific discovered tools
        # This prevents context overflow from servers with many tools (e.g., GitHub has 823 tools)
        original_tools = self._tools
        filtered_tools = {}

        for server, tools in original_tools.items():
            if server == "meta-mcp":
                # Always include all meta-mcp tools
                filtered_tools[server] = tools
            elif server in self.discovered_tools:
                # Only include the specific tools that were discovered/used from this server
                discovered_tool_names = self.discovered_tools[server]
                filtered_tools[server] = [
                    tool for tool in tools
                    if tool.name in discovered_tool_names
                ]

        self._tools = filtered_tools

        try:
            # Call parent to build prompt with filtered tools
            return super()._build_prompt(question)
        finally:
            # Restore original tools (needed for actual tool execution)
            self._tools = original_tools

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

            # Check if this is a Smithery server that requires OAuth
            server_url = config.get("streamable_http", {}).get("url", "")
            is_smithery = "smithery.ai" in server_url

            auth_provider = None
            callback_handler = None

            if is_smithery:
                self._logger.info(f"🔐 Smithery server detected, using OAuth authentication...")
                # Use OAuth for Smithery servers
                from mcpuniverse.mcp.oauth import create_smithery_auth
                import warnings

                # Suppress anyio cancel scope warnings that occur during cleanup
                # These are expected when OAuth servers fail to connect
                warnings.filterwarnings('ignore', message='.*cancel scope.*', category=RuntimeWarning)

                # Remove any API key from URL
                base_url = server_url.split("?")[0]

                auth_provider, callback_handler = create_smithery_auth(
                    server_url=base_url,
                    client_name=f"MCP Universe - {server_name}",
                    redirect_port=8765,
                    timeout=600.0
                )

            # Build client - use the callback_handler in a context manager
            # This ensures proper cleanup even if errors occur
            if callback_handler:
                # Install a custom exception handler to suppress background task errors
                # This is necessary because anyio TaskGroup cleanup can raise RuntimeError
                # in background tasks that can't be caught with try-except
                loop = asyncio.get_event_loop()
                original_exception_handler = loop.get_exception_handler()

                def suppress_cancel_scope_errors(loop, context):
                    """Suppress 'cancel scope in different task' errors during server loading."""
                    exception = context.get('exception')

                    # Helper function to check if exception is cancel scope error
                    def is_cancel_scope_error(exc):
                        if exc is None:
                            return False
                        return isinstance(exc, RuntimeError) and 'cancel scope' in str(exc).lower()

                    # Check for RuntimeError directly
                    if is_cancel_scope_error(exception):
                        self._logger.debug(f"Suppressed background task RuntimeError: {exception}")
                        return

                    # Check for ExceptionGroup/BaseExceptionGroup containing RuntimeError
                    # This handles both Python 3.11+ ExceptionGroup and anyio's BaseExceptionGroup
                    if hasattr(exception, 'exceptions'):
                        # Recursively check all sub-exceptions
                        def has_cancel_scope_error(exc):
                            if is_cancel_scope_error(exc):
                                return True
                            if hasattr(exc, 'exceptions'):
                                return any(has_cancel_scope_error(sub) for sub in exc.exceptions)
                            return False

                        if has_cancel_scope_error(exception):
                            self._logger.debug(f"Suppressed ExceptionGroup with cancel scope error")
                            return

                    # For other exceptions, use original handler or default
                    if original_exception_handler:
                        original_exception_handler(loop, context)
                    else:
                        loop.default_exception_handler(context)

                loop.set_exception_handler(suppress_cancel_scope_errors)

                # CRITICAL: Run server loading in a shielded task to isolate cancel scopes
                # This prevents anyio cancel scopes from polluting the event loop
                async def _load_oauth_server():
                    """Inner function to load server - runs in isolated context."""
                    async with callback_handler:
                        client = await self._mcp_manager.build_client(
                            server_name,
                            transport="streamable_http",
                            auth=auth_provider
                        )
                        self._mcp_clients[server_name] = client

                        # Load tools
                        tools = await client.list_tools()
                        self._tools[server_name] = tools

                        self.loaded_servers.add(server_name)
                        self.dynamically_loaded_servers.add(server_name)
                        self._logger.info(f"✅ Loaded {len(tools)} tools from {server_name}")
                        return (True, "")

                # Use asyncio.shield to protect against cancel scope pollution
                try:
                    result = await asyncio.shield(_load_oauth_server())
                    return result
                except (asyncio.CancelledError, GeneratorExit) as e:
                    # CRITICAL: Catch CancelledError and GeneratorExit here
                    # These occur when MCP SDK's async generators are cancelled during cleanup
                    self._logger.warning(f"⚠ OAuth server loading cancelled for {server_name}: {type(e).__name__}")
                    return (False, f"oauth_cancelled: {type(e).__name__} during OAuth flow")
                except RuntimeError as e:
                    # Catch anyio cancel scope errors that occur during MCP SDK cleanup
                    # These happen when the streamable_http_client's anyio TaskGroup tries to exit
                    if "cancel scope" in str(e).lower():
                        self._logger.warning(f"⚠ OAuth server {server_name} encountered cancel scope error (likely due to HTTP error during connection)")
                        return (False, f"cancel_scope_error: {str(e)[:150]}")
                    else:
                        # Other RuntimeErrors should still be raised
                        raise
                except BaseException as e:
                    # Catch ALL exceptions including system exceptions during cleanup
                    # This is necessary because anyio TaskGroup errors can escape normal exception handling
                    self._logger.warning(f"⚠ OAuth server loading failed for {server_name}: {type(e).__name__} - {str(e)[:200]}")
                    # Don't re-raise - convert to error tuple
                    return (False, f"{type(e).__name__}: {str(e)[:150]}")
                finally:
                    # CRITICAL: Wait for any pending background tasks to complete FIRST
                    # This ensures that any RuntimeError from anyio cleanup is caught
                    # by our custom exception handler (still active at this point)
                    await asyncio.sleep(0.1)  # Give background tasks time to finish

                    # Also explicitly wait for any pending tasks
                    pending = [task for task in asyncio.all_tasks(loop)
                              if not task.done() and task != asyncio.current_task()]
                    if pending:
                        self._logger.debug(f"Waiting for {len(pending)} pending background tasks...")
                        try:
                            await asyncio.wait(pending, timeout=0.5)
                        except Exception as e:
                            self._logger.debug(f"Error waiting for background tasks: {e}")

                    # NOW restore original exception handler (after all tasks are done)
                    loop.set_exception_handler(original_exception_handler)
            else:
                # No OAuth - simpler path
                client = await self._mcp_manager.build_client(
                    server_name,
                    transport="streamable_http",
                    auth=auth_provider
                )
                self._mcp_clients[server_name] = client

                # Load tools
                tools = await client.list_tools()
                self._tools[server_name] = tools

                self.loaded_servers.add(server_name)
                self.dynamically_loaded_servers.add(server_name)
                self._logger.info(f"✅ Loaded {len(tools)} tools from {server_name}")

                return (True, "")

        except asyncio.CancelledError as e:
            # Handle asyncio task cancellation (e.g., from cleanup errors)
            # This prevents the cancellation from propagating and crashing the agent
            self._logger.warning(f"⚠ Server loading cancelled for {server_name} (non-critical)")
            return (False, f"task_cancelled: {str(e)[:100] if str(e) else 'asyncio task was cancelled during server loading'}")
        except RuntimeError as e:
            # Catch anyio cancel scope errors at outer level too
            if "cancel scope" in str(e).lower():
                self._logger.warning(f"⚠ Server {server_name} loading failed due to cancel scope error")
                return (False, f"cancel_scope_error: {str(e)[:150]}")
            else:
                # Other RuntimeErrors - treat as generic errors
                self._logger.warning(f"⚠ RuntimeError loading server {server_name}: {str(e)[:200]}")
                return (False, f"runtime_error: {str(e)[:150]}")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            self._logger.warning(f"⚠ Failed to load server {server_name}: {error_type} - {error_msg[:200]}")

            # Categorize error type
            if "421" in error_msg or "Misdirected Request" in error_msg:
                return (False, f"http_421_misdirected: Server may be misconfigured or OAuth issue")
            elif "403" in error_msg or "Forbidden" in error_msg:
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
                tool_name = tool_call.get("tool")

                # Track this tool as discovered (for prompt filtering)
                if server_name and tool_name and server_name != "meta-mcp":
                    if server_name not in self.discovered_tools:
                        self.discovered_tools[server_name] = set()
                    self.discovered_tools[server_name].add(tool_name)
                    self._logger.debug(f"📝 Tracked discovered tool: {server_name}/{tool_name}")

                # Try to load server if not already loaded
                if server_name not in self.loaded_servers:
                    self._logger.info(f"📡 Server {server_name} not loaded, attempting dynamic load...")
                    success, error_reason = await self._load_server_on_demand(server_name)
                    server_loaded_dynamically = success
                    load_error_reason = error_reason if not success else None

                    if not success:
                        # Return error as tool result with helpful guidance
                        from mcp.types import CallToolResult, TextContent

                        # Provide helpful error message based on error type
                        if "421" in error_reason or "misdirected" in error_reason.lower():
                            error_msg = (
                                f"⚠ Server '{server_name}' is currently unavailable (HTTP 421 - Misdirected Request). "
                                f"This Smithery server may be misconfigured or experiencing OAuth issues.\n\n"
                                f"💡 Suggestion: Try using search_tools again with a similar query to find alternative tools "
                                f"from different servers that can accomplish the same task."
                            )
                        elif "cancelled" in error_reason.lower():
                            error_msg = (
                                f"⚠ Server '{server_name}' connection was cancelled during loading. "
                                f"This may be due to network issues or server timeout.\n\n"
                                f"💡 Suggestion: Search for alternative tools using search_tools with the same capability description."
                            )
                        else:
                            error_msg = (
                                f"⚠ Server '{server_name}' could not be loaded. Reason: {error_reason}\n\n"
                                f"💡 Suggestion: Use search_tools to find alternative tools from different servers."
                            )

                        self._logger.warning(f"Returning error to agent for failed server load: {server_name}")
                        return CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=error_msg
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
            elif not self.enable_compression and result_str and len(result_str) > 50000:
                # When compression is disabled but result is too large, truncate with warning
                from mcp.types import CallToolResult, TextContent
                truncated_str = (
                    f"⚠️ Tool result is too long ({len(result_str)} characters, exceeds 50000 limit).\n"
                    f"Compression is disabled. Showing first 1000 characters only.\n\n"
                    f"--- Result Preview (first 1000 chars) ---\n"
                    f"{result_str[:1000]}\n"
                    f"--- Truncated ({len(result_str) - 1000} characters omitted) ---"
                )
                result = CallToolResult(
                    content=[TextContent(type="text", text=truncated_str)]
                )
                result_str = truncated_str
                self._logger.warning(
                    f"⚠️  Tool result truncated: {len(original_result_str)} → {len(truncated_str)} chars "
                    f"(compression disabled, result too large)"
                )

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
                except asyncio.CancelledError:
                    # Suppress CancelledError during cleanup
                    self._logger.warning(f"Cleanup cancelled for client {server_name} (suppressed)")
                except Exception as e:
                    self._logger.warning(f"Error closing client {server_name}: {e}")

        # Close the parent agent's clients
        try:
            await super().cleanup()
        except asyncio.CancelledError:
            # Suppress CancelledError during parent cleanup
            self._logger.warning("Parent cleanup cancelled (suppressed)")
        except AttributeError:
            # Parent class might not have cleanup method
            pass
        except Exception as e:
            self._logger.warning(f"Error during parent cleanup: {e}")

        self._logger.info("✓ Cleanup complete")
