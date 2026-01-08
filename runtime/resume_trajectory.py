#!/usr/bin/env python3
"""
Resume Timed-Out Trajectories

This script resumes goal-oriented trajectories that failed due to LLM timeouts
or other recoverable errors. It loads the partial trajectory, reconstructs
the conversation state, and continues from where it left off.

Features:
- UUID-based tracking: Each UUID is tracked until it has a timeout-free trajectory
- Automatic checkpoint/recovery: If interrupted, re-running continues from where left off
- --until-clean mode: Keeps retrying until ALL UUIDs have timeout-free trajectories
- Tracks progress in a state file (.resume_state.json) in the target directory

Usage:
    # Resume a single trajectory
    python runtime/resume_trajectory.py /path/to/trajectory_xxx.json \
        --model openai/gpt-5.2 \
        --user-model openai/gpt-5.2

    # Resume all timeout trajectories in a directory
    python runtime/resume_trajectory.py /path/to/trajectories/pass@3 \
        --batch --category llm_timeout \
        --model openai/gpt-5.2 \
        --max-concurrent 3

    # Keep retrying until ALL UUIDs have timeout-free trajectories
    python runtime/resume_trajectory.py /path/to/trajectories/pass@3 \
        --batch --until-clean \
        --model openai/gpt-5.2 \
        --max-concurrent 3

    # Dry run to see what would be resumed
    python runtime/resume_trajectory.py /path/to/trajectories/pass@3 \
        --batch --dry-run

    # Clear previous state and start fresh
    python runtime/resume_trajectory.py /path/to/trajectories/pass@3 \
        --batch --reset-state
"""
from __future__ import annotations

import sys
import json
import asyncio
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not installed. Run 'pip install tqdm' for progress bars.")

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ORCHESTRATOR_DIR))

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.agent.dynamic_react import DynamicReActAgent
from dotenv import load_dotenv

# Import from run_goaloriented_agent
from runtime.run_goaloriented_agent import (
    AGENT_INSTRUCTION,
    USER_PERSONAS,
    SubgoalTracker,
    GoalOrientedUser,
    GoalOrientedController,
    GoalTurn,
    GoalTrajectory,
)


# ============================================================================
# State Management for Recovery
# ============================================================================

class ResumeStateManager:
    """
    Manages checkpoint state for batch resume operations.
    Allows recovery if the script is interrupted mid-batch.
    """

    STATE_FILENAME = ".resume_state.json"

    def __init__(self, directory: Path):
        self.directory = directory
        self.state_file = directory / self.STATE_FILENAME
        self.state: Dict[str, Any] = {
            "started_at": None,
            "completed": {},  # path -> {"status": str, "output": str, "timestamp": str}
            "in_progress": {},  # path -> {"started_at": str}
            "failed": {},  # path -> {"error": str, "timestamp": str}
            "skipped": {},  # path -> {"reason": str, "timestamp": str}
        }
        self._load()

    def _load(self):
        """Load existing state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    saved = json.load(f)
                self.state.update(saved)
                print(f"📂 Loaded resume state: {len(self.state['completed'])} completed, "
                      f"{len(self.state['failed'])} failed, {len(self.state['skipped'])} skipped")
            except Exception as e:
                print(f"⚠️  Could not load state file: {e}")

    def _save(self):
        """Save current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save state file: {e}")

    def reset(self):
        """Clear all state and start fresh."""
        self.state = {
            "started_at": datetime.now().isoformat(),
            "completed": {},
            "in_progress": {},
            "failed": {},
            "skipped": {},
        }
        self._save()
        print("🔄 State reset - starting fresh")

    def is_done(self, path: str) -> bool:
        """Check if a trajectory has already been processed."""
        return (path in self.state["completed"] or
                path in self.state["skipped"])

    def mark_started(self, path: str):
        """Mark a trajectory as in-progress."""
        self.state["in_progress"][path] = {
            "started_at": datetime.now().isoformat()
        }
        # Remove from failed if retrying
        if path in self.state["failed"]:
            del self.state["failed"][path]
        self._save()

    def mark_completed(self, path: str, output_path: str, goal_completion: float = 0.0, timeout_free: bool = False):
        """Mark a trajectory as successfully completed."""
        if path in self.state["in_progress"]:
            del self.state["in_progress"][path]
        self.state["completed"][path] = {
            "status": "success",
            "output": output_path,
            "goal_completion": goal_completion,
            "timeout_free": timeout_free,
            "timestamp": datetime.now().isoformat()
        }
        self._save()

    def mark_failed(self, path: str, error: str):
        """Mark a trajectory as failed."""
        if path in self.state["in_progress"]:
            del self.state["in_progress"][path]
        self.state["failed"][path] = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self._save()

    def mark_skipped(self, path: str, reason: str):
        """Mark a trajectory as skipped."""
        if path in self.state["in_progress"]:
            del self.state["in_progress"][path]
        self.state["skipped"][path] = {
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self._save()

    def get_pending(self, all_paths: List[Path]) -> List[Path]:
        """Filter to only paths that haven't been processed yet."""
        pending = []
        for p in all_paths:
            path_str = str(p)
            if not self.is_done(path_str):
                pending.append(p)
        return pending

    def get_summary(self) -> Dict[str, int]:
        """Get summary counts."""
        return {
            "completed": len(self.state["completed"]),
            "failed": len(self.state["failed"]),
            "skipped": len(self.state["skipped"]),
            "in_progress": len(self.state["in_progress"]),
        }

    def cleanup_in_progress(self):
        """Move any in_progress items to failed (interrupted)."""
        for path in list(self.state["in_progress"].keys()):
            self.state["failed"][path] = {
                "error": "Interrupted (was in progress when script stopped)",
                "timestamp": datetime.now().isoformat()
            }
            del self.state["in_progress"][path]
        self._save()


# ============================================================================
# Trajectory Analysis
# ============================================================================

LLM_ERROR_PATTERNS = {
    'llm_timeout': [
        'llm request timed out after multiple retries',
        'llm request timed out',
    ],
    'llm_rate_limit': [
        '"code":429',
        'status code 429',
        'rate limit exceeded',
    ],
    'llm_insufficient_funds': [
        '"code":402',
        'status code 402',
        'insufficient credits',
        'payment required',
    ],
}


def detect_error_type(trajectory_data: Dict) -> Optional[str]:
    """Detect the type of error in a trajectory."""
    turns = trajectory_data.get("turns", [])

    for turn in turns:
        agent_response = turn.get("agent_response", "")
        if isinstance(agent_response, dict):
            agent_response = json.dumps(agent_response)
        response_lower = str(agent_response).lower()

        for error_type, patterns in LLM_ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in response_lower:
                    return error_type

    return None


def is_trajectory_timeout_free(filepath: Path) -> bool:
    """
    Check if a trajectory file is free of timeout errors.
    Returns True if NO timeout is detected.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check metadata flag first (new trajectories set this when timeout occurs)
        metadata = data.get('metadata', {})
        if metadata.get('timeout_occurred', False):
            return False

        # Also check turn content for legacy trajectories
        for turn in data.get('turns', []):
            agent_response = turn.get('agent_response', '')
            if isinstance(agent_response, dict):
                agent_response = json.dumps(agent_response)
            response_lower = str(agent_response).lower()

            # Check for timeout patterns
            if 'timed out' in response_lower:
                return False

        return True
    except Exception:
        return False


def extract_uuid_from_filepath(filepath: Path) -> Optional[str]:
    """Extract UUID from trajectory filename."""
    # Filename format: trajectory_UUID_TIMESTAMP[_resumed].json
    name = filepath.stem  # Remove .json
    parts = name.split('_')
    if len(parts) >= 2 and parts[0] == 'trajectory':
        return parts[1]
    return None


def get_trajectory_progress(filepath: Path) -> Tuple[int, float, int]:
    """
    Score a trajectory by its progress.

    Returns tuple of (successful_turns, goal_completion_rate, total_tool_calls)
    Higher values = more progress made before timeout.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        turns = data.get('turns', [])
        metadata = data.get('metadata', {})
        goal_completion = metadata.get('goal_completion_rate', 0.0)

        # Count successful turns (before timeout)
        successful_turns = 0
        total_tool_calls = 0

        for turn in turns:
            agent_response = turn.get('agent_response', '')
            if isinstance(agent_response, dict):
                agent_response = json.dumps(agent_response)
            response_lower = str(agent_response).lower()

            # Count tool calls in this turn
            tool_calls = turn.get('tool_calls', [])
            total_tool_calls += len(tool_calls)

            # Check if this turn timed out
            if 'timed out' in response_lower:
                break  # Stop counting after timeout

            # Count as successful if it has a meaningful response
            if agent_response and len(str(agent_response)) > 50:
                successful_turns += 1

        return (successful_turns, goal_completion, total_tool_calls)
    except Exception:
        return (0, 0.0, 0)


def select_best_trajectory_for_resume(filepaths: List[Path]) -> Optional[Path]:
    """
    Select the best trajectory to resume from.

    Prioritizes by:
    1. Most successful turns completed
    2. Highest goal completion rate
    3. Most tool calls made
    """
    if not filepaths:
        return None

    # Score each file
    scored = []
    for fp in filepaths:
        turns, goal_rate, tools = get_trajectory_progress(fp)
        # Create a composite score: prioritize turns, then goal rate, then tools
        score = (turns * 1000) + (goal_rate * 100) + tools
        scored.append((score, turns, goal_rate, fp))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0]
    return best[3]  # Return the filepath


def get_uuid_status(directory: Path) -> Dict[str, Dict]:
    """
    Analyze all trajectories and return status by UUID.

    Returns dict with:
    - uuid -> {
        'has_timeout_free': bool,
        'timeout_free_files': List[Path],
        'timed_out_files': List[Path],
        'best_for_resume': Optional[Path]  # Best file to resume from (most progress)
      }
    """
    uuid_status = {}

    for filepath in directory.glob("**/trajectory_*.json"):
        uuid = extract_uuid_from_filepath(filepath)
        if not uuid:
            continue

        if uuid not in uuid_status:
            uuid_status[uuid] = {
                'has_timeout_free': False,
                'timeout_free_files': [],
                'timed_out_files': [],
                'best_for_resume': None,
            }

        if is_trajectory_timeout_free(filepath):
            uuid_status[uuid]['has_timeout_free'] = True
            uuid_status[uuid]['timeout_free_files'].append(filepath)
        else:
            uuid_status[uuid]['timed_out_files'].append(filepath)

    # Find the BEST trajectory to resume from for each UUID
    # (the one with most progress, not just the most recent)
    for uuid, status in uuid_status.items():
        if status['timed_out_files']:
            status['best_for_resume'] = select_best_trajectory_for_resume(status['timed_out_files'])

    return uuid_status


def get_uuids_needing_work(directory: Path) -> List[Dict]:
    """
    Get list of UUIDs that don't have any timeout-free trajectory.

    Returns list of dicts with:
    - uuid: str
    - resume_from: Path  # The best file to resume from (most progress)
    - progress: Tuple  # (turns, goal_rate, tools) for display
    """
    uuid_status = get_uuid_status(directory)

    needs_work = []
    for uuid, status in uuid_status.items():
        if not status['has_timeout_free'] and status['best_for_resume']:
            progress = get_trajectory_progress(status['best_for_resume'])
            needs_work.append({
                'uuid': uuid,
                'resume_from': status['best_for_resume'],
                'progress': progress,  # (turns, goal_rate, tools)
            })

    return needs_work


def get_uuid_status_single_dir(directory: Path) -> Dict[str, Dict]:
    """
    Analyze trajectories in a SINGLE directory (no recursion into subdirs).
    Used for per-pass mode where each pass@N directory is treated separately.
    """
    uuid_status = {}

    # Only look at direct children, not recursive
    for filepath in directory.glob("trajectory_*.json"):
        uuid = extract_uuid_from_filepath(filepath)
        if not uuid:
            continue

        if uuid not in uuid_status:
            uuid_status[uuid] = {
                'has_timeout_free': False,
                'timeout_free_files': [],
                'timed_out_files': [],
                'best_for_resume': None,
            }

        if is_trajectory_timeout_free(filepath):
            uuid_status[uuid]['has_timeout_free'] = True
            uuid_status[uuid]['timeout_free_files'].append(filepath)
        else:
            uuid_status[uuid]['timed_out_files'].append(filepath)

    # Find the BEST trajectory to resume from for each UUID
    for uuid, status in uuid_status.items():
        if status['timed_out_files']:
            status['best_for_resume'] = select_best_trajectory_for_resume(status['timed_out_files'])

    return uuid_status


def get_all_uuids_from_parent(parent_dir: Path) -> set:
    """Get all unique UUIDs from all pass@N subdirectories."""
    all_uuids = set()
    for pass_dir in parent_dir.iterdir():
        if pass_dir.is_dir() and pass_dir.name.startswith('pass@'):
            for filepath in pass_dir.glob("trajectory_*.json"):
                uuid = extract_uuid_from_filepath(filepath)
                if uuid:
                    all_uuids.add(uuid)
    return all_uuids


def get_per_pass_status(parent_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Analyze status per pass directory.

    Returns:
    {
        'pass@1': {uuid: {has_timeout_free, best_for_resume, ...}, ...},
        'pass@2': {...},
        'pass@3': {...},
    }
    """
    per_pass = {}
    for pass_dir in sorted(parent_dir.iterdir()):
        if pass_dir.is_dir() and pass_dir.name.startswith('pass@'):
            per_pass[pass_dir.name] = get_uuid_status_single_dir(pass_dir)
    return per_pass


def get_uuids_needing_work_per_pass(parent_dir: Path, all_uuids: set) -> Dict[str, List[Dict]]:
    """
    For each pass directory, get UUIDs that need a timeout-free trajectory.

    Returns:
    {
        'pass@1': [{'uuid': ..., 'resume_from': ..., 'progress': ...}, ...],
        'pass@2': [...],
        'pass@3': [...],
    }
    """
    per_pass_status = get_per_pass_status(parent_dir)
    needs_work = {}

    for pass_name, uuid_status in per_pass_status.items():
        pass_dir = parent_dir / pass_name
        needs_work[pass_name] = []

        for uuid in all_uuids:
            status = uuid_status.get(uuid, {})
            has_clean = status.get('has_timeout_free', False)

            if not has_clean:
                # Need to create/resume a trajectory for this UUID in this pass
                best = status.get('best_for_resume')
                if best:
                    # Has a timed-out trajectory to resume from
                    progress = get_trajectory_progress(best)
                    needs_work[pass_name].append({
                        'uuid': uuid,
                        'resume_from': best,
                        'progress': progress,
                        'pass_dir': pass_dir,
                    })
                else:
                    # No trajectory at all for this UUID in this pass
                    # We need to find the best from ANY pass to use as base
                    # For now, mark as needing fresh start
                    needs_work[pass_name].append({
                        'uuid': uuid,
                        'resume_from': None,  # Will need to find from other passes
                        'progress': (0, 0.0, 0),
                        'pass_dir': pass_dir,
                    })

    return needs_work


def find_last_successful_turn(trajectory_data: Dict) -> int:
    """
    Find the index of the last turn that completed successfully (before timeout).
    Returns -1 if no successful turns.
    """
    turns = trajectory_data.get("turns", [])
    last_successful = -1

    for i, turn in enumerate(turns):
        agent_response = turn.get("agent_response", "")
        if isinstance(agent_response, dict):
            agent_response = json.dumps(agent_response)
        response_lower = str(agent_response).lower()

        # Check if this turn has an error
        has_error = False
        for error_type, patterns in LLM_ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in response_lower:
                    has_error = True
                    break
            if has_error:
                break

        if not has_error and agent_response and len(agent_response) > 50:
            # This turn looks successful (has substantial response, no error patterns)
            last_successful = i

    return last_successful


def extract_conversation_history(trajectory_data: Dict, up_to_turn: int) -> List[Dict]:
    """Extract conversation history up to a specific turn for context injection."""
    turns = trajectory_data.get("turns", [])
    history = []

    for i, turn in enumerate(turns):
        if i > up_to_turn:
            break
        history.append({
            "role": "user",
            "content": turn.get("query", "")
        })
        history.append({
            "role": "assistant",
            "content": turn.get("agent_response", "")
        })

    return history


def extract_subgoal_state(trajectory_data: Dict, up_to_turn: int) -> Dict:
    """Extract subgoal completion state up to a specific turn."""
    metadata = trajectory_data.get("metadata", {})
    turns = trajectory_data.get("turns", [])

    all_sub_goals = metadata.get("sub_goals", [])

    # Collect all completed subgoals up to the specified turn
    completed = set()
    for i, turn in enumerate(turns):
        if i > up_to_turn:
            break
        for sg in turn.get("completed_sub_goals", []):
            completed.add(sg)

    remaining = [sg for sg in all_sub_goals if sg not in completed]

    return {
        "all_sub_goals": all_sub_goals,
        "completed": list(completed),
        "remaining": remaining,
        "progress": len(completed) / len(all_sub_goals) if all_sub_goals else 0.0
    }


# ============================================================================
# Resume Controller
# ============================================================================

class ResumeController:
    """Controller that resumes a trajectory from a checkpoint."""

    def __init__(
        self,
        trajectory_path: Path,
        agent_model: str,
        user_model: str,
        max_iterations: int = 60,
        max_additional_turns: int = 10,
    ):
        self.trajectory_path = trajectory_path
        self.agent_model = agent_model
        self.user_model = user_model
        self.max_iterations = max_iterations
        self.max_additional_turns = max_additional_turns

        # Will be loaded
        self.original_data: Dict = {}
        self.last_successful_turn: int = -1
        self.subgoal_state: Dict = {}

    def load_trajectory(self, force: bool = False) -> bool:
        """Load and analyze the trajectory file.

        Args:
            force: If True, skip error detection and force load for resume.
                   Used in --until-clean mode where we know the UUID needs work.
        """
        try:
            with open(self.trajectory_path, 'r') as f:
                self.original_data = json.load(f)

            error_type = detect_error_type(self.original_data)
            if not error_type and not force:
                print(f"⚠️  No recoverable error detected in {self.trajectory_path.name}")
                return False

            self.last_successful_turn = find_last_successful_turn(self.original_data)
            if self.last_successful_turn < 0:
                print(f"⚠️  No successful turns found in {self.trajectory_path.name}")
                # Still allow resuming from turn 0
                self.last_successful_turn = -1

            self.subgoal_state = extract_subgoal_state(
                self.original_data,
                self.last_successful_turn
            )

            print(f"✓ Loaded trajectory: {self.trajectory_path.name}")
            print(f"  Error type: {error_type if error_type else 'none (forced resume)'}")
            print(f"  Last successful turn: {self.last_successful_turn + 1}")
            print(f"  Subgoal progress: {self.subgoal_state['progress']:.0%}")
            print(f"  Remaining subgoals: {len(self.subgoal_state['remaining'])}")

            return True

        except Exception as e:
            print(f"✗ Failed to load {self.trajectory_path}: {e}")
            return False

    async def resume(self) -> Optional[Dict]:
        """Resume the trajectory and return the updated trajectory data."""
        metadata = self.original_data.get("metadata", {})
        seed_query = metadata.get("seed_query", "")
        user_goal = metadata.get("user_goal", seed_query)
        persona_name = metadata.get("user_persona", "curious_researcher")
        query_uuid = metadata.get("uuid", "unknown")
        constraints = metadata.get("constraints", [])

        # Determine the next query to send
        turns = self.original_data.get("turns", [])
        if self.last_successful_turn >= 0 and self.last_successful_turn < len(turns):
            last_turn = turns[self.last_successful_turn]
            # Use the follow-up from last successful turn, or remaining subgoals
            next_query = last_turn.get("follow_up_intent")
            if not next_query and self.subgoal_state["remaining"]:
                # Construct a query asking about remaining subgoals
                remaining_str = ", ".join(self.subgoal_state["remaining"][:3])
                next_query = f"Please continue with the remaining tasks: {remaining_str}"
        else:
            # No successful turns, restart from seed query
            next_query = seed_query

        print(f"\n{'='*70}")
        print(f"RESUMING TRAJECTORY")
        print(f"{'='*70}")
        print(f"UUID: {query_uuid}")
        print(f"Starting from turn: {self.last_successful_turn + 2}")
        print(f"Next query: {next_query[:100]}...")
        print(f"{'='*70}\n")

        try:
            # Initialize components
            load_dotenv(str(ORCHESTRATOR_DIR / ".env"))

            mcp_manager = MCPManager()
            model_manager = ModelManager()

            # Add Meta-MCP server
            meta_mcp_config = {
                "stdio": {
                    "command": "python",
                    "args": [str(PROJECT_ROOT / "tool_retrieval_index" / "server.py")],
                }
            }
            mcp_manager.add_server_config("meta-mcp", meta_mcp_config)

            # Build server configs
            server_list = json.loads(
                (PROJECT_ROOT / "MCP_INFO_MGR/mcp_data/working/remote_servers.json").read_text()
            )

            fmp_config = os.environ.get("FMP_CONFIG_BASE64", "")
            server_specific_configs = {
                "@imbenrabi/financial-modeling-prep-mcp-server": fmp_config,
                "@vijitdaroch/financial-modeling-prep-mcp-server": fmp_config,
                "@hollaugo/financial-research-mcp": fmp_config,
                "@hollaugo/financial-research-mcp-server": fmp_config,
                "@Parichay-Pothepalli/financial-research-mcp": fmp_config,
            }

            def build_server_url(server_name: str) -> str:
                base_url = f"https://server.smithery.ai/{server_name}"
                config_b64 = server_specific_configs.get(server_name, "")
                if config_b64:
                    return f"{base_url}?config={config_b64}"
                return base_url

            all_server_configs = {
                server: {
                    "streamable_http": {"url": build_server_url(server), "headers": {}},
                    "env": {}
                }
                for server in server_list
            }

            # Create LLMs
            agent_llm = model_manager.build_model("openrouter", config={"model_name": self.agent_model, "timeout": 600})
            user_llm = model_manager.build_model("openrouter", config={"model_name": self.user_model, "timeout": 600})

            # Create agent
            react_config = {
                "name": "resume-react-agent",
                "instruction": AGENT_INSTRUCTION,
                "max_iterations": self.max_iterations,
                "summarize_tool_response": "auto",
                "summarize_threshold": 100000,
            }

            agent = DynamicReActAgent(
                mcp_manager=mcp_manager,
                llm=agent_llm,
                server_configs=all_server_configs,
                config=react_config,
            )
            await agent.initialize(mcp_servers=[{"name": "meta-mcp"}])

            # Create subgoal tracker with pre-loaded state
            subgoal_tracker = SubgoalTracker(
                llm=user_llm,
                query=user_goal,
                constraints=constraints
            )
            # Pre-populate the tracker state
            subgoal_tracker.sub_goals = self.subgoal_state["all_sub_goals"]
            subgoal_tracker.completed = self.subgoal_state["completed"]
            subgoal_tracker.remaining = self.subgoal_state["remaining"]

            # Create user
            max_turns = USER_PERSONAS[persona_name]["max_turns"]
            goal_user = GoalOrientedUser(
                llm=user_llm,
                persona_name=persona_name,
                query=user_goal,
                subgoal_tracker=subgoal_tracker
            )

            # Create controller
            controller = GoalOrientedController(
                agent=agent,
                goal_oriented_user=goal_user,
                subgoal_tracker=subgoal_tracker,
                max_turns=min(max_turns, self.max_additional_turns),
                query_uuid=query_uuid,
                enable_bonus_questions=False
            )

            # Inject previous successful turns into controller
            existing_turns = []
            for i, turn_data in enumerate(turns):
                if i > self.last_successful_turn:
                    break
                existing_turns.append(GoalTurn(
                    turn_number=turn_data.get("turn_number", i + 1),
                    query=turn_data.get("query", ""),
                    agent_response=turn_data.get("agent_response", ""),
                    tool_calls=turn_data.get("tool_calls", []),
                    reasoning_trace=turn_data.get("reasoning_trace", []),
                    available_servers=turn_data.get("available_servers", []),
                    available_tool_count=turn_data.get("available_tool_count", 0),
                    completed_sub_goals=turn_data.get("completed_sub_goals", []),
                    remaining_sub_goals=turn_data.get("remaining_sub_goals", []),
                    goal_progress=turn_data.get("goal_progress", 0.0),
                    constraints_violated=turn_data.get("constraints_violated", []),
                    constraint_satisfaction_rate=turn_data.get("constraint_satisfaction_rate", 1.0),
                    user_decision=turn_data.get("user_decision", "CONTINUE"),
                    termination_reason=turn_data.get("termination_reason"),
                    satisfaction_level=turn_data.get("satisfaction_level", 0.5),
                    user_reasoning=turn_data.get("user_reasoning", ""),
                    follow_up_intent=turn_data.get("follow_up_intent")
                ))
            controller.turns = existing_turns

            # Run the resumed conversation
            trajectory = await controller.run_conversation(next_query)

            # Cleanup
            try:
                await agent.cleanup()
            except Exception as e:
                print(f"⚠️  Warning during cleanup: {e}")

            # Check if the trajectory contains a timeout - if so, DON'T return it
            result_dict = trajectory.to_dict()
            for turn in result_dict.get("turns", []):
                agent_response = turn.get("agent_response", "")
                if isinstance(agent_response, dict):
                    agent_response = json.dumps(agent_response)
                if "timed out" in str(agent_response).lower():
                    print(f"⚠️  Resume produced a timeout - NOT saving this trajectory")
                    return None  # Return None so it's marked as failed and will be retried

            return result_dict

        except Exception as e:
            print(f"✗ Resume failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Batch Processing
# ============================================================================

def find_problematic_trajectories(
    directory: Path,
    category: Optional[str] = None
) -> List[Path]:
    """Find all trajectories with recoverable errors."""
    problematic = []

    for filepath in directory.glob("**/trajectory_*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            error_type = detect_error_type(data)
            if error_type:
                if category is None or error_type == category:
                    problematic.append(filepath)

        except Exception as e:
            print(f"⚠️  Could not analyze {filepath}: {e}")

    return problematic


async def resume_single(
    trajectory_path: Path,
    args,
    semaphore: asyncio.Semaphore,
    state_manager: Optional[ResumeStateManager] = None,
    output_dir_override: Optional[Path] = None,
    force: bool = False
) -> Dict:
    """Resume a single trajectory with semaphore control and state tracking.

    Args:
        force: If True, bypass error detection and force resume.
               Used in --until-clean mode where UUID was selected as needing work.
    """
    async with semaphore:
        path_str = str(trajectory_path)

        # Mark as started
        if state_manager:
            state_manager.mark_started(path_str)

        controller = ResumeController(
            trajectory_path=trajectory_path,
            agent_model=args.model,
            user_model=args.user_model,
            max_iterations=args.max_iterations,
            max_additional_turns=args.max_additional_turns,
        )

        if not controller.load_trajectory(force=force):
            reason = "no recoverable error"
            if state_manager:
                state_manager.mark_skipped(path_str, reason)
            return {
                "path": path_str,
                "status": "skipped",
                "reason": reason
            }

        try:
            result = await controller.resume()

            if result:
                # Save the resumed trajectory
                # Priority: output_dir_override > args.output_dir > trajectory_path.parent
                if output_dir_override:
                    output_dir = output_dir_override
                elif args.output_dir:
                    output_dir = Path(args.output_dir)
                else:
                    output_dir = trajectory_path.parent

                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate new filename with "resumed" suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                uuid = result.get("metadata", {}).get("uuid", "unknown")
                output_file = output_dir / f"trajectory_{uuid}_{timestamp}_resumed.json"

                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                print(f"💾 Saved resumed trajectory: {output_file}")

                # Check if the output is actually timeout-free
                output_is_clean = is_trajectory_timeout_free(output_file)
                if output_is_clean:
                    print(f"✅ Output is TIMEOUT-FREE!")
                else:
                    print(f"⚠️  Output still has timeout (will retry if --until-clean)")

                # Optionally delete or rename the original
                if args.archive_original:
                    archive_path = trajectory_path.with_suffix('.json.archived')
                    trajectory_path.rename(archive_path)
                    print(f"📦 Archived original: {archive_path}")

                goal_completion = result.get("metadata", {}).get("goal_completion_rate", 0)

                # Mark as completed in state (include whether it's clean)
                if state_manager:
                    state_manager.mark_completed(path_str, str(output_file), goal_completion, output_is_clean)

                return {
                    "path": path_str,
                    "status": "success",
                    "output": str(output_file),
                    "goal_completion": goal_completion,
                    "timeout_free": output_is_clean
                }
            else:
                error = "resume execution failed or produced timeout (will retry)"
                if state_manager:
                    state_manager.mark_failed(path_str, error)
                return {
                    "path": path_str,
                    "status": "failed",
                    "reason": error
                }

        except Exception as e:
            error = str(e)
            if state_manager:
                state_manager.mark_failed(path_str, error)
            return {
                "path": path_str,
                "status": "failed",
                "reason": error
            }


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Resume timed-out goal-oriented trajectories"
    )
    parser.add_argument(
        "path",
        help="Path to trajectory file or directory"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all problematic trajectories in directory"
    )
    parser.add_argument(
        "--category",
        choices=["llm_timeout", "llm_rate_limit", "llm_insufficient_funds"],
        default="llm_timeout",
        help="Error category to resume (default: llm_timeout)"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="Model for agent (OpenRouter format)"
    )
    parser.add_argument(
        "--user-model",
        default="openai/gpt-5.2",
        help="Model for simulated user (OpenRouter format)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=60,
        help="Max agent reasoning iterations per turn"
    )
    parser.add_argument(
        "--max-additional-turns",
        type=int,
        default=10,
        help="Max additional turns after resuming"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent resume operations (batch mode)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for resumed trajectories (default: same as input)"
    )
    parser.add_argument(
        "--archive-original",
        action="store_true",
        help="Archive original trajectory files after successful resume"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be resumed without actually doing it"
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Clear previous state and start fresh (batch mode)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also retry previously failed trajectories (batch mode)"
    )
    parser.add_argument(
        "--until-clean",
        action="store_true",
        help="Keep retrying until ALL UUIDs have at least one timeout-free trajectory"
    )
    parser.add_argument(
        "--per-pass",
        action="store_true",
        help="Ensure EACH pass@N directory has a timeout-free trajectory for EVERY UUID"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum number of retry rounds in --until-clean mode (default: 10)"
    )

    args = parser.parse_args()

    target_path = Path(args.path)

    if args.batch or target_path.is_dir():
        # Batch mode
        if not target_path.is_dir():
            parser.error(f"Batch mode requires a directory: {target_path}")

        # Initialize state manager
        state_manager = ResumeStateManager(target_path)

        if args.reset_state:
            state_manager.reset()

        # Clean up any interrupted jobs from previous run
        state_manager.cleanup_in_progress()

        # If retrying failed, clear failed state
        if args.retry_failed:
            print(f"🔄 Retrying {len(state_manager.state['failed'])} previously failed trajectories")
            state_manager.state["failed"] = {}
            state_manager._save()

        if args.per_pass:
            # ================================================================
            # --per-pass mode: Ensure EACH pass@N has a clean trajectory for EVERY UUID
            # ================================================================
            print(f"\n{'='*70}")
            print("PER-PASS MODE: Ensuring each pass@N has timeout-free trajectory for every UUID")
            print(f"Max rounds: {args.max_rounds}")
            print(f"{'='*70}\n")

            # Check if target has pass@N subdirectories
            pass_dirs = [d for d in target_path.iterdir() if d.is_dir() and d.name.startswith('pass@')]
            if not pass_dirs:
                print(f"❌ No pass@N directories found in {target_path}")
                print("   Expected directories like: pass@1, pass@2, pass@3")
                return

            print(f"Found {len(pass_dirs)} pass directories: {', '.join(d.name for d in sorted(pass_dirs))}")

            # Get all UUIDs across all passes
            all_uuids = get_all_uuids_from_parent(target_path)
            print(f"Total unique UUIDs: {len(all_uuids)}")

            total_processed = 0
            total_success = 0
            total_timeout_free = 0

            for round_num in range(1, args.max_rounds + 1):
                # Get status per pass
                per_pass_needs = get_uuids_needing_work_per_pass(target_path, all_uuids)
                per_pass_status = get_per_pass_status(target_path)

                # Count totals
                total_needed = sum(len(items) for items in per_pass_needs.values())

                print(f"\n{'='*70}")
                print(f"ROUND {round_num}/{args.max_rounds}")
                print(f"{'='*70}")

                # Show status per pass
                all_clean = True
                for pass_name in sorted(per_pass_status.keys()):
                    uuid_status = per_pass_status[pass_name]
                    clean_count = sum(1 for s in uuid_status.values() if s.get('has_timeout_free', False))
                    missing = len(all_uuids) - clean_count
                    print(f"  {pass_name}: {clean_count}/{len(all_uuids)} clean ({missing} missing)")
                    if missing > 0:
                        all_clean = False

                if all_clean:
                    print(f"\n🎉 ALL PASSES HAVE TIMEOUT-FREE TRAJECTORIES FOR ALL UUIDs!")
                    break

                if args.dry_run:
                    print("\nDRY RUN - Would resume/create:")
                    for pass_name in sorted(per_pass_needs.keys()):
                        items = per_pass_needs[pass_name]
                        if items:
                            print(f"\n  {pass_name} ({len(items)} UUIDs):")
                            # Only show items that have a resume_from (can actually be resumed)
                            resumable = [i for i in items if i.get('resume_from')]
                            missing = [i for i in items if not i.get('resume_from')]
                            if resumable:
                                for item in resumable[:3]:
                                    turns, goal_rate, tools = item['progress']
                                    print(f"    - {item['uuid'][:12]}... from {item['resume_from'].name}")
                                    print(f"      Progress: {turns} turns, {goal_rate:.0%} goal, {tools} tools")
                                if len(resumable) > 3:
                                    print(f"    ... and {len(resumable) - 3} more resumable")
                            if missing:
                                print(f"    ⚠️  {len(missing)} UUIDs have NO trajectory in this pass (need fresh run)")
                    return

                # Process each pass sequentially
                for pass_name in sorted(per_pass_needs.keys()):
                    items = per_pass_needs[pass_name]
                    # Only process items that have a resume_from path
                    resumable_items = [i for i in items if i.get('resume_from')]

                    if not resumable_items:
                        continue

                    pass_dir = target_path / pass_name
                    print(f"\n📂 Processing {pass_name}: {len(resumable_items)} UUIDs to resume...")

                    semaphore = asyncio.Semaphore(args.max_concurrent)

                    try:
                        if TQDM_AVAILABLE:
                            tasks = [
                                resume_single(item['resume_from'], args, semaphore, state_manager,
                                              output_dir_override=pass_dir, force=True)
                                for item in resumable_items
                            ]
                            results = []
                            with tqdm(total=len(tasks), desc=pass_name, unit="uuid") as pbar:
                                for coro in asyncio.as_completed(tasks):
                                    result = await coro
                                    results.append(result)
                                    if isinstance(result, dict):
                                        status = "✓" if result.get('timeout_free') else "○" if result.get('status') == 'success' else "✗"
                                        pbar.set_postfix_str(f"Last: {status}")
                                    pbar.update(1)
                        else:
                            tasks = [
                                resume_single(item['resume_from'], args, semaphore, state_manager,
                                              output_dir_override=pass_dir, force=True)
                                for item in resumable_items
                            ]
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Interrupted! Progress has been saved.")
                        print(f"Re-run the same command to continue.")
                        return

                    # Count results
                    success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
                    timeout_free = sum(1 for r in results if isinstance(r, dict) and r.get("timeout_free", False))

                    total_processed += len(resumable_items)
                    total_success += success
                    total_timeout_free += timeout_free

                    print(f"  {pass_name} results: {success} success, {timeout_free} timeout-free")

                # Check if we made progress
                if total_timeout_free == 0 and total_success == 0:
                    print(f"\n⚠️  No progress made in this round.")

            # Final summary
            per_pass_status = get_per_pass_status(target_path)
            print(f"\n{'='*70}")
            print("FINAL SUMMARY")
            print(f"{'='*70}")
            print(f"Total operations: {total_processed}")
            print(f"Successful resumes: {total_success}")
            print(f"New timeout-free: {total_timeout_free}")
            print(f"\nPer-pass status:")
            for pass_name in sorted(per_pass_status.keys()):
                uuid_status = per_pass_status[pass_name]
                clean_count = sum(1 for s in uuid_status.values() if s.get('has_timeout_free', False))
                print(f"  {pass_name}: {clean_count}/{len(all_uuids)} timeout-free")
            print(f"{'='*70}\n")

        elif args.until_clean:
            # ================================================================
            # --until-clean mode: Loop until ALL UUIDs have timeout-free trajectories
            # ================================================================
            print(f"\n{'='*70}")
            print("UNTIL-CLEAN MODE: Will keep retrying until all UUIDs are timeout-free")
            print(f"Max rounds: {args.max_rounds}")
            print(f"{'='*70}\n")

            total_processed = 0
            total_success = 0
            total_timeout_free = 0

            for round_num in range(1, args.max_rounds + 1):
                # Get UUIDs that still need work
                needs_work = get_uuids_needing_work(target_path)
                uuid_status = get_uuid_status(target_path)

                # Count current status
                total_uuids = len(uuid_status)
                clean_uuids = sum(1 for s in uuid_status.values() if s['has_timeout_free'])

                print(f"\n{'='*70}")
                print(f"ROUND {round_num}/{args.max_rounds}")
                print(f"{'='*70}")
                print(f"UUID Status: {clean_uuids}/{total_uuids} are timeout-free ({100*clean_uuids/total_uuids:.1f}%)")
                print(f"UUIDs needing work: {len(needs_work)}")

                if not needs_work:
                    print(f"\n🎉 ALL UUIDs HAVE TIMEOUT-FREE TRAJECTORIES!")
                    print(f"Total UUIDs: {total_uuids}")
                    print(f"All clean: {clean_uuids}")
                    break

                if args.dry_run:
                    print("\nDRY RUN - Would resume these UUIDs (sorted by progress):")
                    # Sort by progress for display
                    sorted_work = sorted(needs_work, key=lambda x: x['progress'], reverse=True)
                    for item in sorted_work:
                        turns, goal_rate, tools = item['progress']
                        print(f"  - UUID: {item['uuid']}")
                        print(f"    From: {item['resume_from'].name}")
                        print(f"    Progress: {turns} turns, {goal_rate:.0%} goal, {tools} tools")
                    return

                # Process all UUIDs needing work with tqdm progress bar
                semaphore = asyncio.Semaphore(args.max_concurrent)

                print(f"\nStarting {len(needs_work)} resume operations (max {args.max_concurrent} concurrent)...")
                print("Progress is saved after each completion - safe to interrupt with Ctrl+C\n")

                try:
                    if TQDM_AVAILABLE:
                        # Use tqdm for progress tracking
                        # force=True because these UUIDs were selected as needing work
                        tasks = [
                            resume_single(item['resume_from'], args, semaphore, state_manager, force=True)
                            for item in needs_work
                        ]
                        results = []
                        with tqdm(total=len(tasks), desc=f"Round {round_num}", unit="uuid") as pbar:
                            for coro in asyncio.as_completed(tasks):
                                result = await coro
                                results.append(result)
                                # Update progress bar with result info
                                if isinstance(result, dict):
                                    status = "✓" if result.get('timeout_free') else "○" if result.get('status') == 'success' else "✗"
                                    pbar.set_postfix_str(f"Last: {status}")
                                pbar.update(1)
                    else:
                        # Fallback without tqdm
                        # force=True because these UUIDs were selected as needing work
                        tasks = [
                            resume_single(item['resume_from'], args, semaphore, state_manager, force=True)
                            for item in needs_work
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted! Progress has been saved.")
                    print(f"Re-run the same command to continue from where you left off.")
                    return

                # Count results
                success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
                timeout_free = sum(1 for r in results if isinstance(r, dict) and r.get("timeout_free", False))

                total_processed += len(needs_work)
                total_success += success
                total_timeout_free += timeout_free

                print(f"\nRound {round_num} results:")
                print(f"  Processed: {len(needs_work)}")
                print(f"  Success: {success}")
                print(f"  Timeout-free: {timeout_free}")

                # Check if we made progress
                if timeout_free == 0 and success == 0:
                    print(f"\n⚠️  No progress made in this round. Consider:")
                    print(f"    - Increasing timeout (currently 180s)")
                    print(f"    - Checking if LLM provider is having issues")
                    print(f"    - Running during off-peak hours")

            else:
                # Max rounds reached
                needs_work = get_uuids_needing_work(target_path)
                print(f"\n⚠️  Max rounds ({args.max_rounds}) reached.")
                print(f"    Still {len(needs_work)} UUIDs need timeout-free trajectories.")
                print(f"    Re-run with --until-clean to continue.")

            # Final summary
            uuid_status = get_uuid_status(target_path)
            total_uuids = len(uuid_status)
            clean_uuids = sum(1 for s in uuid_status.values() if s['has_timeout_free'])

            print(f"\n{'='*70}")
            print("FINAL SUMMARY")
            print(f"{'='*70}")
            print(f"Total operations: {total_processed}")
            print(f"Successful resumes: {total_success}")
            print(f"New timeout-free: {total_timeout_free}")
            print(f"\nUUID Status: {clean_uuids}/{total_uuids} are timeout-free ({100*clean_uuids/total_uuids:.1f}%)")
            print(f"{'='*70}\n")

        else:
            # ================================================================
            # Original single-pass batch mode
            # ================================================================
            print(f"\n🔍 Scanning for {args.category} trajectories in: {target_path}")
            all_problematic = find_problematic_trajectories(target_path, args.category)

            # Filter to only pending (not already completed/skipped)
            problematic = state_manager.get_pending(all_problematic)

            print(f"Found {len(all_problematic)} total problematic trajectories")
            print(f"Already processed: {len(all_problematic) - len(problematic)}")
            print(f"Remaining to resume: {len(problematic)}\n")

            if args.dry_run:
                print("DRY RUN - Would resume:")
                for p in problematic:
                    print(f"  - {p}")
                return

            if not problematic:
                print("No trajectories to resume (all already processed).")
                summary = state_manager.get_summary()
                print(f"\nPrevious run summary:")
                print(f"  Completed: {summary['completed']}")
                print(f"  Failed: {summary['failed']}")
                print(f"  Skipped: {summary['skipped']}")
                return

            # Process in parallel with tqdm progress bar
            semaphore = asyncio.Semaphore(args.max_concurrent)

            print(f"Starting {len(problematic)} resume operations (max {args.max_concurrent} concurrent)...")
            print("Progress is saved after each completion - safe to interrupt with Ctrl+C\n")

            try:
                if TQDM_AVAILABLE:
                    tasks = [
                        resume_single(p, args, semaphore, state_manager)
                        for p in problematic
                    ]
                    results = []
                    with tqdm(total=len(tasks), desc="Resuming", unit="traj") as pbar:
                        for coro in asyncio.as_completed(tasks):
                            result = await coro
                            results.append(result)
                            if isinstance(result, dict):
                                status = "✓" if result.get('timeout_free') else "○" if result.get('status') == 'success' else "✗"
                                pbar.set_postfix_str(f"Last: {status}")
                            pbar.update(1)
                else:
                    tasks = [
                        resume_single(p, args, semaphore, state_manager)
                        for p in problematic
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted! Progress has been saved.")
                summary = state_manager.get_summary()
                print(f"  Completed so far: {summary['completed']}")
                print(f"  Failed: {summary['failed']}")
                print(f"  In progress (will retry): {summary['in_progress']}")
                print(f"\nRe-run the same command to continue from where you left off.")
                return

            # Summary
            success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
            failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
            skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
            errors = sum(1 for r in results if isinstance(r, Exception))
            timeout_free = sum(1 for r in results if isinstance(r, dict) and r.get("timeout_free", False))

            # Get overall summary including previous runs
            overall = state_manager.get_summary()

            # Also show UUID-level status
            uuid_status = get_uuid_status(target_path)
            total_uuids = len(uuid_status)
            clean_uuids = sum(1 for s in uuid_status.values() if s['has_timeout_free'])

            print(f"\n{'='*70}")
            print("BATCH RESUME SUMMARY")
            print(f"{'='*70}")
            print(f"This run:")
            print(f"  Processed: {len(problematic)}")
            print(f"  Success: {success}")
            print(f"  Timeout-free: {timeout_free}")
            print(f"  Failed: {failed}")
            print(f"  Skipped: {skipped}")
            print(f"  Errors: {errors}")
            print(f"\nOverall (including previous runs):")
            print(f"  Total completed: {overall['completed']}")
            print(f"  Total failed: {overall['failed']}")
            print(f"  Total skipped: {overall['skipped']}")
            print(f"\nUUID Status: {clean_uuids}/{total_uuids} are timeout-free ({100*clean_uuids/total_uuids:.1f}%)")
            print(f"{'='*70}")
            print(f"State saved to: {state_manager.state_file}")
            print(f"\n💡 TIP: Use --until-clean to keep retrying until all UUIDs are timeout-free")
            print(f"{'='*70}\n")

    else:
        # Single file mode
        if not target_path.exists():
            parser.error(f"File not found: {target_path}")

        if args.dry_run:
            print(f"DRY RUN - Would resume: {target_path}")
            return

        controller = ResumeController(
            trajectory_path=target_path,
            agent_model=args.model,
            user_model=args.user_model,
            max_iterations=args.max_iterations,
            max_additional_turns=args.max_additional_turns,
        )

        if not controller.load_trajectory():
            print("Cannot resume this trajectory.")
            sys.exit(1)

        result = await controller.resume()

        if result:
            # Save
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = target_path.parent

            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid = result.get("metadata", {}).get("uuid", "unknown")
            output_file = output_dir / f"trajectory_{uuid}_{timestamp}_resumed.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Saved resumed trajectory: {output_file}")

            if args.archive_original:
                archive_path = target_path.with_suffix('.json.archived')
                target_path.rename(archive_path)
                print(f"📦 Archived original: {archive_path}")

            print(f"\n✅ Resume completed successfully!")
            print(f"   Goal completion: {result.get('metadata', {}).get('goal_completion_rate', 0):.0%}")
        else:
            print("\n✗ Resume failed")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
