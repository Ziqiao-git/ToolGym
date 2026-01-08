#!/usr/bin/env python3
"""
Emergency Tool Call Interceptor (Legacy Module)

This module is deprecated. Please use state_controller.py instead.

The StateController provides a more comprehensive implementation with:
- Tool-level control (timeouts, rate limits, unavailability)
- State-level control (delayed/corrupted results, session timeouts)
- Constraint-level control (runtime constraint changes)

For backward compatibility, this module re-exports the legacy classes
from state_controller.py.

Usage (deprecated):
    from emergency_interceptor import EmergencyInterceptor, InterceptionStrategy

Usage (recommended):
    from state_controller import StateController, ToolControlPolicy, ToolFailureStrategy
"""

import warnings

# Re-export from new module for backward compatibility
from state_controller import (
    # Legacy compatibility classes
    EmergencyInterceptor,
    InterceptionStrategy,

    # New classes (for gradual migration)
    StateController,
    ControlPolicy,
    ToolControlPolicy,
    StateControlPolicy,
    ConstraintControlPolicy,
    ToolFailureStrategy,
    StateControlStrategy,
    ConstraintStrategy,

    # Factory functions
    create_tool_failure_controller,
    create_state_manipulation_controller,
    create_constraint_controller,
    create_comprehensive_controller,
)

# Emit deprecation warning when this module is imported directly
warnings.warn(
    "emergency_interceptor is deprecated. Use state_controller instead.\n"
    "Example: from state_controller import StateController, ToolFailureStrategy",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    # Legacy
    "EmergencyInterceptor",
    "InterceptionStrategy",

    # New
    "StateController",
    "ControlPolicy",
    "ToolControlPolicy",
    "StateControlPolicy",
    "ConstraintControlPolicy",
    "ToolFailureStrategy",
    "StateControlStrategy",
    "ConstraintStrategy",
    "create_tool_failure_controller",
    "create_state_manipulation_controller",
    "create_constraint_controller",
    "create_comprehensive_controller",
]
