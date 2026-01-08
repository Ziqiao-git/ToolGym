"""
Backward compatibility shim for state_controller.

This module re-exports from toolgym.state_controller for backward compatibility
with existing code in runtime/.
"""

import sys
from pathlib import Path

# Add toolgym to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Re-export everything from new location
from toolgym.state_controller import (
    StateController,
    ControlPolicy,
    ToolControlPolicy,
    StateControlPolicy,
    ConstraintControlPolicy,
    ToolFailureStrategy,
    StateControlStrategy,
    ConstraintStrategy,
    create_tool_failure_controller,
    create_state_manipulation_controller,
    create_constraint_controller,
    create_comprehensive_controller,
    EmergencyInterceptor,
    InterceptionStrategy,
)

__all__ = [
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
    "EmergencyInterceptor",
    "InterceptionStrategy",
]
