#!/usr/bin/env python3
"""
Tool Testing Script for MCP Servers

This script tests all tools from the tool_descriptions.ndjson file by:
1. Loading tool descriptions from NDJSON format
2. Connecting to each MCP server
3. Invoking each tool with generated test parameters (simple or LLM-based)
4. Tracking success/failure status
5. Optionally using LLM to evaluate response quality
6. Generating a comprehensive test report

Usage:
    python test_all_tools.py [OPTIONS]

Options:
    --server SERVER_NAME        Test only this server (qualified name)
    --tool TOOL_NAME           Test only this tool (requires --server)
    --limit N                  Test only first N servers
    --output OUTPUT_FILE       Output file for results (default: tool_test_results.json)
    --use-llm                  Use LLM to generate realistic test arguments
    --evaluate-with-llm        Use LLM to evaluate response quality (implies --use-llm)
    --evaluate-existing FILE   Evaluate existing test results with LLM (no re-testing)
    --model MODEL_NAME         LLM model to use (default: from OPENAI_MODEL env or gpt-4)
    --judge-model MODEL_NAME   Judge model for evaluation (default: from JUDGE_MODEL env)

Testing Modes:
    1. Basic Mode (default): Fast technical testing with simple generated arguments
    2. LLM Mode (--use-llm): LLM generates realistic test arguments
    3. LLM Evaluation Mode (--evaluate-with-llm): LLM generates args AND evaluates responses
    4. Evaluate Existing Mode (--evaluate-existing): Evaluate existing results with LLM

Examples:
    # Fast technical test of all tools
    python test_all_tools.py

    # LLM-based functional test with evaluation
    python test_all_tools.py --evaluate-with-llm

    # Test specific server with LLM-generated arguments
    python test_all_tools.py --server exa --use-llm

    # Test first 10 servers with full LLM evaluation
    python test_all_tools.py --limit 10 --evaluate-with-llm

    # Evaluate existing results without re-testing
    python test_all_tools.py --evaluate-existing tool_probe_result_llm_trial.json --output evaluated.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack
import argparse
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
ORCHESTRATOR_DIR = PROJECT_ROOT.parent / "Orchestrator"
sys.path.insert(0, str(ORCHESTRATOR_DIR))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcpuniverse.mcp.oauth import create_smithery_auth, reset_shared_auth

load_dotenv()

# Rate limiting
import time
from collections import deque

class RateLimiter:
    """Simple rate limiter to avoid API throttling."""
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.timestamps = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self.lock:
            now = time.time()
            # Remove timestamps older than 1 minute
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()

            # If at limit, wait until oldest request is > 1 minute old
            if len(self.timestamps) >= self.max_requests:
                sleep_time = 60 - (now - self.timestamps[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time.time()
                    # Clean up again after sleep
                    while self.timestamps and now - self.timestamps[0] > 60:
                        self.timestamps.popleft()

            self.timestamps.append(now)


# LLM Prompts for testing
ARGUMENT_GENERATION_PROMPT = """You are a tool testing assistant. Generate realistic test arguments for an MCP tool.

Tool Name: {tool_name}
Tool Description: {description}
Input Schema: {input_schema}

Generate realistic test arguments that would demonstrate the tool's functionality. The arguments should be:
1. Valid according to the input schema
2. Realistic and meaningful (not just "test" or dummy values)
3. Likely to produce a successful response

Only include required parameters unless optional ones are clearly beneficial.

Return ONLY a JSON object with the test arguments. No explanation or markdown formatting.

Example output format:
{{"query": "latest news about AI", "limit": 5}}"""


RESPONSE_EVALUATION_PROMPT = """You are evaluating whether a tool is TECHNICALLY FUNCTIONAL (not result quality).

Tool Name: {tool_name}
Tool Description: {description}
Test Arguments: {arguments}
Tool Response: {response}

Determine if the tool is USABLE (executed without technical errors), NOT whether results are high quality.

Mark as FAILURE (success: false) ONLY if:
- HTTP errors (404, 500, etc.) in response
- Exceptions or error messages
- "Tool not configured" or "setup required"
- Completely empty response when data is clearly expected
- Connection/timeout errors

Mark as SUCCESS (success: true) if:
- Tool executed and returned ANY data (even if suboptimal/irrelevant)
- Search returned results (even if not perfectly relevant)
- Tool returned "no results found" status (tool works, just no data)
- Response has valid structure with empty/null values (tool works)

Focus on TECHNICAL ERRORS, not result quality or relevance.

Return a JSON object:
{{
  "success": true/false,
  "execution_score": 0-10 (10 = no errors, 0 = exceptions/errors),
  "relevance_score": 0-10 (always 10 if no errors - we don't judge quality),
  "completeness_score": 0-10 (always 10 if got any response - we don't judge quality),
  "quality_score": 0-10 (always 10 if no errors - we don't judge quality),
  "overall_score": 0-1 (1.0 if usable, 0.0 if broken),
  "explanation": "brief explanation focusing on technical errors only",
  "issues": ["list", "of", "technical", "errors", "only"]
}}

Return ONLY the JSON object. No markdown formatting."""


class ToolTester:
    """Handles testing of MCP tools."""

    def __init__(self, tool_descriptions_path: str,
                 use_llm: bool = False, evaluate_with_llm: bool = False,
                 model: Optional[str] = None, judge_model: Optional[str] = None):
        """
        Initialize the ToolTester.

        Args:
            tool_descriptions_path: Path to tool_descriptions.ndjson
            use_llm: Whether to use LLM for generating test arguments
            evaluate_with_llm: Whether to use LLM for evaluating responses
            model: LLM model name for argument generation
            judge_model: Judge model name for response evaluation
        """
        self.tool_descriptions_path = tool_descriptions_path
        self.tool_descriptions = []
        self.results = []

        # LLM settings
        self.use_llm = use_llm or evaluate_with_llm  # evaluate_with_llm implies use_llm
        self.evaluate_with_llm = evaluate_with_llm
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL", self.model)

        # Initialize OpenAI client if using LLM
        if self.use_llm:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY required for LLM mode")
            self.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            # Rate limiter: 30 requests per minute to avoid throttling
            self.rate_limiter = RateLimiter(max_requests_per_minute=30)

        # Server URLs will be built on-demand
        self.server_urls = {}

        # Track current connection resources for cleanup
        self._current_callback_handler = None
        self._current_transport_ctx = None
        self._current_session = None

    def load_data(self):
        """Load tool descriptions and build server URLs."""
        # Load tool descriptions
        with open(self.tool_descriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Only include entries with status='ok' (successfully fetched tools)
                    if entry.get('status') == 'ok':
                        self.tool_descriptions.append(entry)
        print(f"Loaded {len(self.tool_descriptions)} servers with working tools")

        # Build server URLs for OAuth connections
        for entry in self.tool_descriptions:
            server_name = entry.get("qualifiedName")
            # Use the URL from the entry if available, otherwise construct it
            url = entry.get("url") or f"https://server.smithery.ai/{server_name}"
            self.server_urls[server_name] = url

        print(f"Prepared {len(self.server_urls)} server URLs for OAuth connections")

    def generate_simple_test_arguments(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simple test arguments based on the tool's input schema.
        Used for fast technical testing.

        Args:
            input_schema: JSON schema for tool input

        Returns:
            Dictionary of test arguments
        """
        if not input_schema or "properties" not in input_schema:
            return {}

        args = {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for param_name, param_spec in properties.items():
            param_type = param_spec.get("type")

            # Only generate required parameters to minimize errors
            if param_name not in required:
                continue

            # Generate test values based on type
            if param_type == "string":
                # Use description hints if available
                desc = param_spec.get("description", "").lower()
                if "query" in desc or "search" in desc:
                    args[param_name] = "test query"
                elif "url" in desc or "link" in desc:
                    args[param_name] = "https://example.com"
                elif "path" in desc or "file" in desc:
                    args[param_name] = "/tmp/test.txt"
                elif "email" in desc:
                    args[param_name] = "test@example.com"
                elif "name" in desc:
                    args[param_name] = "test"
                else:
                    args[param_name] = "test"

            elif param_type == "number" or param_type == "integer":
                args[param_name] = 1

            elif param_type == "boolean":
                args[param_name] = False

            elif param_type == "array":
                args[param_name] = []

            elif param_type == "object":
                args[param_name] = {}

        return args

    async def generate_llm_test_arguments(self, tool_name: str, description: str,
                                   input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to generate realistic test arguments.

        Args:
            tool_name: Name of the tool
            description: Tool description
            input_schema: JSON schema for tool input

        Returns:
            Dictionary of test arguments
        """
        prompt = ARGUMENT_GENERATION_PROMPT.format(
            tool_name=tool_name,
            description=description,
            input_schema=json.dumps(input_schema, indent=2)
        )

        try:
            # Rate limit before making API call
            await self.rate_limiter.acquire()

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a tool testing assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Try to extract JSON from markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            args = json.loads(content)
            print(f"    LLM-generated args: {json.dumps(args)}")
            return args

        except Exception as e:
            print(f"    [WARNING] LLM argument generation failed: {e}")
            print(f"    Falling back to simple argument generation")
            return self.generate_simple_test_arguments(input_schema)

    async def evaluate_response_with_llm(self, tool_name: str, description: str,
                                  arguments: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """
        Use LLM to evaluate the quality of a tool response.

        Args:
            tool_name: Name of the tool
            description: Tool description
            arguments: Test arguments used
            response: Tool response to evaluate

        Returns:
            Dictionary with evaluation scores and explanation
        """
        # Convert response to string if needed
        if hasattr(response, 'model_dump'):
            response_str = json.dumps(response.model_dump(mode="json"), indent=2)
        else:
            response_str = str(response)

        # Truncate very long responses
        if len(response_str) > 5000:
            response_str = response_str[:5000] + "\n... (truncated)"

        prompt = RESPONSE_EVALUATION_PROMPT.format(
            tool_name=tool_name,
            description=description,
            arguments=json.dumps(arguments, indent=2),
            response=response_str
        )

        try:
            # Rate limit before making API call
            await self.rate_limiter.acquire()

            response = await self.llm_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are a strict tool evaluation judge. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Try to extract JSON from markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            evaluation = json.loads(content)
            return evaluation

        except Exception as e:
            print(f"    [WARNING] LLM evaluation failed: {e}")
            return {
                "success": False,
                "execution_score": 0,
                "relevance_score": 0,
                "completeness_score": 0,
                "quality_score": 0,
                "overall_score": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "issues": ["LLM evaluation error"]
            }

    async def connect_to_server(self, server_name: str):
        """
        Connect to an MCP server using OAuth authentication.

        Uses the same pattern as fetch_new_tools.py for proper OAuth flow.

        Args:
            server_name: Qualified name of the server (e.g., 'exa', '@user/server')

        Returns:
            Tuple of (session, cleanup_func) if successful, (None, None) otherwise
        """
        url = self.server_urls.get(server_name)
        if not url:
            print(f"  [ERROR] No URL found for server {server_name}")
            return None, None

        # Optional: override token cache location to persist across runs/hosts
        storage_dir = os.getenv("SMITHERY_TOKEN_DIR")

        try:
            # Create OAuth provider for this server
            auth_provider, callback_handler = create_smithery_auth(
                server_url=url,
                client_name="MCP Tool Tester",
                storage_dir=storage_dir,
            )

            # Store references for cleanup
            self._current_callback_handler = callback_handler
            self._current_transport_ctx = None
            self._current_session = None

            # Enter callback handler context
            await callback_handler.__aenter__()

            # Connect with OAuth
            transport_ctx = streamablehttp_client(url=url, auth=auth_provider)
            self._current_transport_ctx = transport_ctx

            try:
                read, write, _ = await asyncio.wait_for(
                    transport_ctx.__aenter__(),
                    timeout=30
                )
            except asyncio.TimeoutError:
                print(f"  [ERROR] Transport connection timeout")
                await self._cleanup_current_connection()
                return None, None

            session = ClientSession(read, write)
            self._current_session = session
            await session.__aenter__()

            try:
                await asyncio.wait_for(session.initialize(), timeout=30)
            except asyncio.TimeoutError:
                print(f"  [ERROR] Session initialization timeout")
                await self._cleanup_current_connection()
                return None, None

            # Create cleanup function
            _callback = callback_handler
            _transport = transport_ctx
            _session = session

            async def cleanup():
                try:
                    await _session.__aexit__(None, None, None)
                except:
                    pass
                try:
                    await _transport.__aexit__(None, None, None)
                except:
                    pass
                try:
                    await _callback.__aexit__(None, None, None)
                except:
                    pass

            return session, cleanup

        except asyncio.CancelledError:
            print(f"  [ERROR] Connection cancelled")
            await self._cleanup_current_connection()
            return None, None
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                print(f"  [ERROR] 401 Unauthorized")
            elif "404" in error_msg:
                print(f"  [ERROR] 404 Not Found")
            else:
                short_msg = error_msg.split('\n')[0][:80]
                print(f"  [ERROR] {short_msg}")
            await self._cleanup_current_connection()
            return None, None

    async def _cleanup_current_connection(self):
        """Cleanup current connection resources."""
        if hasattr(self, '_current_session') and self._current_session:
            try:
                await self._current_session.__aexit__(None, None, None)
            except:
                pass
        if hasattr(self, '_current_transport_ctx') and self._current_transport_ctx:
            try:
                await self._current_transport_ctx.__aexit__(None, None, None)
            except:
                pass
        if hasattr(self, '_current_callback_handler') and self._current_callback_handler:
            try:
                await self._current_callback_handler.__aexit__(None, None, None)
            except:
                pass
        self._current_session = None
        self._current_transport_ctx = None
        self._current_callback_handler = None

    async def test_tool(self, server_name: str, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single tool.

        Args:
            server_name: Name of the server
            tool: Tool description dictionary

        Returns:
            Dictionary with test results
        """
        tool_name = tool.get("name", "unknown")
        tool_description = tool.get("description", "")
        print(f"\n  Testing tool: {tool_name}")

        result = {
            "server": server_name,
            "tool": tool_name,
            "description": tool_description,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "error": None,
            "response": None,
            "test_arguments": None,
            "test_mode": "llm" if self.use_llm else "simple"
        }

        # Add evaluation fields if using LLM evaluation
        if self.evaluate_with_llm:
            result["llm_evaluation"] = None

        # Connect to server
        session, cleanup = await self.connect_to_server(server_name)
        if session is None:
            result["status"] = "connection_failed"
            result["error"] = "Failed to connect to server"
            return result

        try:
            # Generate test arguments
            input_schema = tool.get("inputSchema", {})

            if self.use_llm:
                print(f"    Generating LLM-based test arguments...")
                test_args = await self.generate_llm_test_arguments(tool_name, tool_description, input_schema)
            else:
                test_args = self.generate_simple_test_arguments(input_schema)
                print(f"    Arguments: {json.dumps(test_args)}")

            result["test_arguments"] = test_args

            # Execute tool with timeout
            try:
                tool_result = await asyncio.wait_for(
                    session.call_tool(tool_name, test_args),
                    timeout=30.0  # 30 second timeout
                )

                # Convert result to dictionary if it's a Pydantic model
                if hasattr(tool_result, 'model_dump'):
                    response_data = tool_result.model_dump(mode="json")
                else:
                    response_data = str(tool_result)

                result["response"] = response_data
                result["status"] = "success"

                # Evaluate with LLM if requested
                if self.evaluate_with_llm:
                    print(f"    [SUCCESS] Tool executed, evaluating with LLM...")
                    evaluation = await self.evaluate_response_with_llm(
                        tool_name, tool_description, test_args, tool_result
                    )
                    result["llm_evaluation"] = evaluation

                    # Update status based on LLM evaluation
                    if not evaluation.get("success", False):
                        result["status"] = "success_but_poor_quality"

                    print(f"    [EVALUATED] Score: {evaluation.get('overall_score', 0):.2f}, "
                          f"Issues: {len(evaluation.get('issues', []))}")
                else:
                    print(f"    [SUCCESS] Tool executed successfully")

            except asyncio.TimeoutError:
                result["status"] = "timeout"
                result["error"] = "Tool execution timed out (30s)"
                print(f"    [TIMEOUT] Tool execution timed out")

            except Exception as e:
                result["status"] = "execution_failed"
                result["error"] = str(e)
                result["error_traceback"] = traceback.format_exc()
                print(f"    [FAILED] {str(e)}")

        finally:
            # Cleanup session
            if cleanup:
                try:
                    await cleanup()
                except Exception:
                    pass

        return result

    async def test_server(self, server_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Test all tools for a specific server.

        Args:
            server_data: Server data from tool_descriptions.ndjson

        Returns:
            List of test results
        """
        server_name = server_data.get("qualifiedName", "unknown")
        tools = server_data.get("tools", [])

        print(f"\n{'='*80}")
        print(f"Testing server: {server_name}")
        print(f"Number of tools: {len(tools)}")
        print(f"{'='*80}")

        results = []
        for tool in tools:
            try:
                result = await self.test_tool(server_name, tool)
                results.append(result)
            except Exception as e:
                # Catch any unexpected errors and record them
                print(f"\n  [CRITICAL ERROR] Unexpected error testing {tool.get('name', 'unknown')}: {str(e)[:100]}")
                results.append({
                    "server": server_name,
                    "tool": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                    "timestamp": datetime.now().isoformat(),
                    "status": "critical_error",
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "test_arguments": None,
                    "response": None,
                    "test_mode": "llm" if self.use_llm else "simple"
                })

            # Small delay between tools to avoid rate limiting
            await asyncio.sleep(1)

        return results

    async def run_tests(self, server_filter: Optional[str] = None,
                       tool_filter: Optional[str] = None,
                       limit: Optional[int] = None,
                       one_tool_per_server: bool = False):
        """
        Run tests on all tools.

        Args:
            server_filter: If provided, only test this server
            tool_filter: If provided, only test this tool
            limit: If provided, only test first N servers
            one_tool_per_server: If True, only test first tool from each server
        """
        self.load_data()

        servers_to_test = self.tool_descriptions

        # Filter by server if specified
        if server_filter:
            servers_to_test = [s for s in servers_to_test
                             if s.get("qualifiedName") == server_filter]
            if not servers_to_test:
                print(f"Error: Server '{server_filter}' not found")
                return

        # Apply limit
        if limit:
            servers_to_test = servers_to_test[:limit]

        print(f"\nStarting tests for {len(servers_to_test)} servers...")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Test each server
        for server_data in servers_to_test:
            try:
                # Filter tools if specified
                if tool_filter:
                    tools = server_data.get("tools", [])
                    server_data["tools"] = [t for t in tools if t.get("name") == tool_filter]
                    if not server_data["tools"]:
                        continue
                elif one_tool_per_server:
                    # Only test the first tool from each server
                    tools = server_data.get("tools", [])
                    if tools:
                        server_data["tools"] = [tools[0]]

                results = await self.test_server(server_data)
                self.results.extend(results)
            except Exception as e:
                # Catch server-level errors and continue
                server_name = server_data.get("qualifiedName", "unknown")
                print(f"\n[SERVER ERROR] Failed to test server {server_name}: {str(e)[:100]}")
                print(f"Continuing with next server...\n")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary statistics."""
        print(f"\n\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")

        total = len(self.results)
        if total == 0:
            print("No tests were run.")
            return

        # Count statuses
        success = len([r for r in self.results if r["status"] == "success"])
        success_poor = len([r for r in self.results if r["status"] == "success_but_poor_quality"])
        failed = len([r for r in self.results if r["status"] == "execution_failed"])
        timeout = len([r for r in self.results if r["status"] == "timeout"])
        connection_failed = len([r for r in self.results if r["status"] == "connection_failed"])
        critical_error = len([r for r in self.results if r["status"] == "critical_error"])

        print(f"\nTest Mode: {'LLM-based' if self.use_llm else 'Simple'}")
        if self.use_llm:
            print(f"Generation Model: {self.model}")
        if self.evaluate_with_llm:
            print(f"Evaluation Model: {self.judge_model}")

        print(f"\nTotal tests: {total}")
        print(f"Successful: {success} ({success/total*100:.1f}%)")
        if success_poor > 0:
            print(f"Successful but poor quality: {success_poor} ({success_poor/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Timeout: {timeout} ({timeout/total*100:.1f}%)")
        print(f"Connection failed: {connection_failed} ({connection_failed/total*100:.1f}%)")
        if critical_error > 0:
            print(f"Critical errors: {critical_error} ({critical_error/total*100:.1f}%)")

        # LLM evaluation statistics
        if self.evaluate_with_llm:
            evaluated_results = [r for r in self.results if r.get("llm_evaluation")]
            if evaluated_results:
                avg_score = sum(r["llm_evaluation"].get("overall_score", 0)
                              for r in evaluated_results) / len(evaluated_results)
                print(f"\nLLM Evaluation Statistics:")
                print(f"Average Quality Score: {avg_score:.3f}")

                # Score distribution
                high_quality = len([r for r in evaluated_results
                                  if r["llm_evaluation"].get("overall_score", 0) >= 0.8])
                medium_quality = len([r for r in evaluated_results
                                    if 0.5 <= r["llm_evaluation"].get("overall_score", 0) < 0.8])
                low_quality = len([r for r in evaluated_results
                                 if r["llm_evaluation"].get("overall_score", 0) < 0.5])

                print(f"High quality (≥0.8): {high_quality} ({high_quality/len(evaluated_results)*100:.1f}%)")
                print(f"Medium quality (0.5-0.8): {medium_quality} ({medium_quality/len(evaluated_results)*100:.1f}%)")
                print(f"Low quality (<0.5): {low_quality} ({low_quality/len(evaluated_results)*100:.1f}%)")

        # Group by server
        servers = {}
        for result in self.results:
            server = result["server"]
            if server not in servers:
                servers[server] = {"success": 0, "failed": 0, "total": 0}
            servers[server]["total"] += 1
            if result["status"] in ["success", "success_but_poor_quality"]:
                servers[server]["success"] += 1
            else:
                servers[server]["failed"] += 1

        print(f"\nServers tested: {len(servers)}")
        print("\nTop 10 servers by success rate:")
        sorted_servers = sorted(servers.items(),
                               key=lambda x: x[1]["success"] / x[1]["total"] if x[1]["total"] > 0 else 0,
                               reverse=True)[:10]
        for server, stats in sorted_servers:
            success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {server}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

    def save_results(self, output_path: str):
        """
        Save test results to a JSON file.

        Args:
            output_path: Path to output file
        """
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "results": self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    async def cleanup_shared_resources(self):
        """Cleanup shared OAuth resources."""
        # Cleanup any remaining connection
        await self._cleanup_current_connection()
        # Reset the singleton OAuth resources
        reset_shared_auth()


async def evaluate_existing_results(args):
    """
    Evaluate existing test results with LLM to find false positives.

    This is much faster than re-running all tests - it just uses LLM to check
    if "successful" responses actually contain errors or useful data.
    """
    print("="*80)
    print("EVALUATING EXISTING RESULTS WITH LLM")
    print("="*80)

    input_file = args.evaluate_existing
    output_file = args.output
    model = args.judge_model or args.model or os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4")

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {model}")
    print("="*80)

    # Load existing results
    print(f"\nLoading results from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    print(f"Loaded {len(results)} test results")

    # Count by status
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nOriginal status distribution:")
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {status}: {count} ({count/len(results)*100:.1f}%)")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY required for LLM evaluation")

    llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    rate_limiter = RateLimiter(max_requests_per_minute=30)

    # Only evaluate successful results
    to_evaluate = [r for r in results if r.get('status') == 'success']
    print(f"\nEvaluating {len(to_evaluate)} successful results for false positives...")

    # Evaluate in batches
    batch_size = 10
    false_positives = []

    for i in range(0, len(to_evaluate), batch_size):
        batch = to_evaluate[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}/{(len(to_evaluate)-1)//batch_size + 1} ({len(batch)} results)...")

        # Evaluate batch concurrently
        tasks = []
        for result in batch:
            task = evaluate_single_result(result, llm_client, rate_limiter, model)
            tasks.append(task)

        evaluations = await asyncio.gather(*tasks)

        # Update results
        for result, evaluation in zip(batch, evaluations):
            result['llm_evaluation'] = evaluation

            if not evaluation.get('is_truly_successful', True):
                result['original_status'] = result['status']
                result['status'] = 'false_positive'
                false_positives.append(f"{result['server']}/{result['tool']}")
                print(f"  ⚠️  FALSE POSITIVE: {result['server']}/{result['tool']}")
                print(f"      {evaluation.get('reasoning', 'Unknown reason')}")

    # Update metadata
    data['metadata'] = data.get('metadata', {})
    data['metadata']['evaluated_with_llm'] = True
    data['metadata']['evaluation_model'] = model
    data['metadata']['evaluation_timestamp'] = datetime.now().isoformat()

    # Save
    print(f"\nSaving evaluated results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Print summary
    new_status_counts = {}
    for r in results:
        status = r['status']
        new_status_counts[status] = new_status_counts.get(status, 0) + 1

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print("\nNew status distribution:")
    for status, count in sorted(new_status_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {status}: {count} ({count/len(results)*100:.1f}%)")

    if false_positives:
        print(f"\n⚠️  Found {len(false_positives)} FALSE POSITIVES")
        print(f"These tools appeared successful but contain errors in responses")

    true_success = new_status_counts.get('success', 0)
    print(f"\n✅ TRUE SUCCESS RATE: {true_success}/{len(results)} ({true_success/len(results)*100:.1f}%)")


async def evaluate_single_result(result, llm_client, rate_limiter, model):
    """Evaluate a single result with LLM."""
    tool_name = result.get('tool', 'unknown')
    description = result.get('description', '')
    arguments = result.get('test_arguments', {})
    response = result.get('response', '')

    # Convert response to string
    if isinstance(response, dict):
        response_str = json.dumps(response, indent=2)
    else:
        response_str = str(response)

    # Truncate very long responses
    if len(response_str) > 3000:
        response_str = response_str[:3000] + "\n... (truncated)"

    prompt = f"""You are evaluating whether a tool is TECHNICALLY FUNCTIONAL (not result quality).

Tool: {tool_name}
Description: {description}
Arguments: {json.dumps(arguments, indent=2)}
Response: {response_str}

Determine if the tool is USABLE (executed without technical errors), NOT whether results are high quality.

Mark as FALSE POSITIVE (not usable) ONLY if:
- HTTP errors (404, 500, etc.)
- Exceptions or error messages in response
- "Tool not configured" or "setup required"
- Completely empty response when data is clearly expected
- Connection/timeout errors in response

Mark as TRUE (usable) if:
- Tool executed and returned ANY data (even if suboptimal/irrelevant)
- Search returned results (even if not perfectly relevant)
- Tool returned "no results found" status (tool works, just no data)
- Response has valid structure with empty/null values (tool works)

Focus on TECHNICAL ERRORS, not result quality or relevance.

Return JSON:
{{
  "is_truly_successful": true/false,
  "confidence": 0-1,
  "reasoning": "brief explanation",
  "detected_issues": []
}}

Return ONLY the JSON object."""

    try:
        await rate_limiter.acquire()

        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a tool evaluator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        return json.loads(content)

    except Exception as e:
        return {
            "is_truly_successful": True,  # Default to true if evaluation fails
            "confidence": 0.0,
            "reasoning": f"Evaluation error: {str(e)}",
            "detected_issues": ["evaluation_failed"]
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all MCP tools from tool_descriptions.ndjson",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", help="Input NDJSON file (default: mcp_data/indexed/tool_descriptions.ndjson)")
    parser.add_argument("--server", help="Test only this server (qualified name)")
    parser.add_argument("--tool", help="Test only this tool (requires --server)")
    parser.add_argument("--limit", type=int, help="Test only first N servers")
    parser.add_argument("--one-tool-per-server", action="store_true",
                       help="Test only the first tool from each server (quick health check)")
    parser.add_argument("--output", default="tool_test_results.json",
                       help="Output file for results (default: tool_test_results.json)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM to generate realistic test arguments")
    parser.add_argument("--evaluate-with-llm", action="store_true",
                       help="Use LLM to evaluate response quality (implies --use-llm)")
    parser.add_argument("--evaluate-existing", metavar="FILE",
                       help="Evaluate existing test results with LLM (no re-testing)")
    parser.add_argument("--model", help="LLM model for argument generation (default: from OPENAI_MODEL env)")
    parser.add_argument("--judge-model", help="Judge model for evaluation (default: from JUDGE_MODEL env)")

    args = parser.parse_args()

    # Validate arguments
    if args.tool and not args.server:
        parser.error("--tool requires --server to be specified")

    # If evaluating existing results, run that mode instead
    if args.evaluate_existing:
        await evaluate_existing_results(args)
        return

    # Set up paths (script is now in MCP_INFO_MGR directory)
    data_dir = PROJECT_ROOT / "mcp_data"
    if args.input:
        tool_descriptions_path = Path(args.input)
    else:
        tool_descriptions_path = data_dir / "indexed" / "tool_descriptions.ndjson"

    # Check file exists
    if not tool_descriptions_path.exists():
        print(f"Error: Tool descriptions not found at {tool_descriptions_path}")
        sys.exit(1)

    # Print configuration
    print("="*80)
    print("MCP TOOL TESTING SCRIPT")
    print("="*80)
    if args.use_llm or args.evaluate_with_llm:
        print(f"Mode: {'LLM Evaluation' if args.evaluate_with_llm else 'LLM Arguments'}")
        print(f"Generation Model: {args.model or os.getenv('OPENAI_MODEL', 'gpt-4')}")
        if args.evaluate_with_llm:
            print(f"Judge Model: {args.judge_model or os.getenv('JUDGE_MODEL', args.model or 'gpt-4')}")
    else:
        print("Mode: Simple/Fast (no LLM)")
    print("="*80)

    # Run tests
    try:
        tester = ToolTester(
            tool_descriptions_path=str(tool_descriptions_path),
            use_llm=args.use_llm,
            evaluate_with_llm=args.evaluate_with_llm,
            model=args.model,
            judge_model=args.judge_model
        )

        await tester.run_tests(
            server_filter=args.server,
            tool_filter=args.tool,
            limit=args.limit,
            one_tool_per_server=args.one_tool_per_server
        )

        # Save results
        tester.save_results(args.output)

        # Cleanup shared resources
        await tester.cleanup_shared_resources()

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nFor LLM mode, ensure you have set one of:")
        print("  - OPENAI_API_KEY and optionally OPENAI_BASE_URL")
        print("  - OPENROUTER_API_KEY and optionally OPENROUTER_BASE_URL")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
