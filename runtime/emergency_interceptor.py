#!/usr/bin/env python3
"""
Emergency Tool Call Interceptor

This module provides non-invasive tool call interception for testing agent
robustness in emergency scenarios (e.g., tool failures, server unavailability).

Usage:
    from emergency_interceptor import EmergencyInterceptor, InterceptionStrategy

    # Strategy 1: Intercept first non-search tool
    interceptor = EmergencyInterceptor(strategy=InterceptionStrategy.FIRST_NON_SEARCH)

    # Strategy 2: Intercept at specific iteration
    interceptor = EmergencyInterceptor(
        strategy=InterceptionStrategy.AT_ITERATION,
        intercept_at_iteration=3
    )

    # Inject into agent
    interceptor.inject(agent)
"""
from enum import Enum
from typing import Dict, Any, Optional, Callable
import json
import logging
from datetime import datetime


class InterceptionStrategy(Enum):
    """Interception strategies for emergency testing."""
    NO_INTERCEPTION = "no_interception"  # No interception (control group)
    FIRST_NON_SEARCH = "first_non_search"  # Intercept first non-search tool call
    RANDOM_20 = "random_20"  # 20% probability to intercept each tool call


class EmergencyInterceptor:
    """
    Non-invasive tool call interceptor using monkey patching.

    This class wraps the agent's call_tool method to intercept and simulate
    tool failures without modifying the original codebase.
    """

    def __init__(
        self,
        strategy: InterceptionStrategy = InterceptionStrategy.FIRST_NON_SEARCH,
        error_message: str = "Error: Tool temporarily unavailable (503 Service Unavailable)",
        exclude_tools: list[str] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize emergency interceptor.

        Args:
            strategy: Interception strategy to use
            error_message: Error message to return when intercepting
            exclude_tools: List of tool names to never intercept (default: ["search_tools"])
            random_seed: Random seed for reproducible random interception (optional)
        """
        self.strategy = strategy
        self.error_message = error_message
        self.exclude_tools = exclude_tools or ["search_tools"]
        self.random_seed = random_seed

        # Internal state
        self._first_non_search_intercepted = False
        self._tool_call_count = 0
        self._non_search_call_count = 0
        self._interception_log = []
        self._original_call_tool = None
        self._logger = logging.getLogger(__name__)

        # Initialize random number generator for random strategy
        if random_seed is not None:
            import random
            random.seed(random_seed)

    def inject(self, agent):
        """
        Inject interceptor into agent by wrapping its call_tool method.

        Args:
            agent: DynamicReActAgent instance to inject into
        """
        # Save original method
        self._original_call_tool = agent.call_tool

        # Create wrapper
        async def intercepted_call_tool(llm_response, tracer=None, callbacks=None):
            return await self._intercept_call_tool(
                agent, llm_response, tracer, callbacks
            )

        # Replace method
        agent.call_tool = intercepted_call_tool
        self._logger.info(f"✅ Emergency interceptor injected with strategy: {self.strategy.value}")

    async def _intercept_call_tool(self, agent, llm_response, tracer=None, callbacks=None):
        """
        Interceptor wrapper for call_tool method.

        This decides whether to intercept or pass through to original method.
        """
        self._tool_call_count += 1

        # Parse tool call to get tool name
        tool_name = self._extract_tool_name(llm_response)

        # Check if this is a search tool (excluded from interception)
        is_search_tool = tool_name in self.exclude_tools

        if not is_search_tool:
            self._non_search_call_count += 1

        # Decide whether to intercept
        should_intercept = self._should_intercept(tool_name, is_search_tool)

        if should_intercept:
            # Mark first_non_search as intercepted (for that strategy)
            if self.strategy == InterceptionStrategy.FIRST_NON_SEARCH:
                self._first_non_search_intercepted = True

            return await self._simulate_failure(
                agent, llm_response, tool_name, tracer, callbacks
            )
        else:
            # Pass through to original method
            return await self._original_call_tool(llm_response, tracer, callbacks)

    def _should_intercept(self, tool_name: Optional[str], is_search_tool: bool) -> bool:
        """
        Determine if current tool call should be intercepted based on strategy.

        Args:
            tool_name: Name of the tool being called
            is_search_tool: Whether this is a search tool (excluded)

        Returns:
            True if should intercept, False otherwise
        """
        # Never intercept search tools
        if is_search_tool:
            return False

        # Strategy 0: No interception (control group)
        if self.strategy == InterceptionStrategy.NO_INTERCEPTION:
            return False

        # Strategy 1: First non-search tool (intercept only once)
        elif self.strategy == InterceptionStrategy.FIRST_NON_SEARCH:
            if self._first_non_search_intercepted:
                return False
            return self._non_search_call_count == 1

        # Strategy 2: Random 20% probability
        elif self.strategy == InterceptionStrategy.RANDOM_20:
            import random
            return random.random() < 0.20

        return False

    def _extract_tool_name(self, llm_response) -> Optional[str]:
        """
        Extract tool name from LLM response.

        Args:
            llm_response: LLM response (str or dict)

        Returns:
            Tool name or None if cannot parse
        """
        try:
            if isinstance(llm_response, str):
                _response = llm_response.strip().strip('`').strip()
                if _response.startswith("json"):
                    _response = _response[4:].strip()
                tool_call = json.loads(_response)
            else:
                tool_call = llm_response

            return tool_call.get("tool")
        except Exception as e:
            self._logger.warning(f"Failed to extract tool name: {e}")
            return None

    async def _simulate_failure(
        self, agent, llm_response, tool_name, tracer, callbacks
    ):
        """
        Simulate tool failure and return error result.

        Args:
            agent: Agent instance
            llm_response: Original LLM response
            tool_name: Name of tool being intercepted
            tracer: Tracer instance
            callbacks: Callback instances

        Returns:
            CallToolResult with error message
        """
        from mcp.types import CallToolResult, TextContent

        # Log interception
        interception_info = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy.value,
            "tool_name": tool_name,
            "tool_call_count": self._tool_call_count,
            "non_search_call_count": self._non_search_call_count,
            "error_message": self.error_message,
        }
        self._interception_log.append(interception_info)

        self._logger.warning(
            f"🚨 INTERCEPTED: {tool_name} (strategy: {self.strategy.value}, "
            f"call #{self._tool_call_count})"
        )

        # Return simulated error
        return CallToolResult(
            content=[TextContent(type="text", text=self.error_message)]
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get interception statistics.

        Returns:
            Dictionary with interception stats
        """
        return {
            "strategy": self.strategy.value,
            "total_tool_calls": self._tool_call_count,
            "non_search_tool_calls": self._non_search_call_count,
            "intercepted": len(self._interception_log) > 0,
            "interception_count": len(self._interception_log),
            "interception_log": self._interception_log,
        }

    def reset(self):
        """Reset interceptor state (for reuse)."""
        self._first_non_search_intercepted = False
        self._tool_call_count = 0
        self._non_search_call_count = 0
        self._interception_log = []

        # Re-seed random if seed was provided
        if self.random_seed is not None:
            import random
            random.seed(self.random_seed)
