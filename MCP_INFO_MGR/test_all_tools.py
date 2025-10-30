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
    --model MODEL_NAME         LLM model to use (default: from OPENAI_MODEL env or gpt-4)
    --judge-model MODEL_NAME   Judge model for evaluation (default: from JUDGE_MODEL env)

Testing Modes:
    1. Basic Mode (default): Fast technical testing with simple generated arguments
    2. LLM Mode (--use-llm): LLM generates realistic test arguments
    3. LLM Evaluation Mode (--evaluate-with-llm): LLM generates args AND evaluates responses

Examples:
    # Fast technical test of all tools
    python test_all_tools.py

    # LLM-based functional test with evaluation
    python test_all_tools.py --evaluate-with-llm

    # Test specific server with LLM-generated arguments
    python test_all_tools.py --server exa --use-llm

    # Test first 10 servers with full LLM evaluation
    python test_all_tools.py --limit 10 --evaluate-with-llm
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
from openai import OpenAI
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.mcp.config import ServerConfig

load_dotenv()


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


RESPONSE_EVALUATION_PROMPT = """You are a tool testing evaluator. Assess whether a tool execution was successful and useful.

Tool Name: {tool_name}
Tool Description: {description}
Test Arguments: {arguments}
Tool Response: {response}

Evaluate the tool execution on these criteria:
1. Execution Success: Did the tool execute without errors?
2. Response Relevance: Is the response relevant to the arguments provided?
3. Response Completeness: Does the response appear complete and useful?
4. Response Quality: Is the response of good quality (not empty, not error messages)?

Return a JSON object with your evaluation:
{{
  "success": true/false,
  "execution_score": 0-10,
  "relevance_score": 0-10,
  "completeness_score": 0-10,
  "quality_score": 0-10,
  "overall_score": 0-1 (average of above scores / 10),
  "explanation": "brief explanation of the evaluation",
  "issues": ["list", "of", "any", "issues", "found"]
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
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL", self.model)

        # Initialize OpenAI client if using LLM
        if self.use_llm:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY required for LLM mode")
            self.llm_client = OpenAI(api_key=api_key, base_url=base_url)

        # MCP Manager will be created after loading data
        self.mcp_manager = None

    def load_data(self):
        """Load tool descriptions and build server configs."""
        # Load tool descriptions
        with open(self.tool_descriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Only include entries with status='ok' (successfully fetched tools)
                    if entry.get('status') == 'ok':
                        self.tool_descriptions.append(entry)
        print(f"Loaded {len(self.tool_descriptions)} servers with working tools")

        # Build server configs for MCP Manager
        smithery_api_key = os.getenv("SMITHERY_API_KEY", "")
        if not smithery_api_key:
            raise ValueError("SMITHERY_API_KEY not found in environment")

        server_configs = {}
        for entry in self.tool_descriptions:
            server_name = entry.get("qualifiedName")
            url = f"https://server.smithery.ai/{server_name}/mcp?api_key={smithery_api_key}"
            server_configs[server_name] = {
                "streamable_http": {
                    "url": url,
                    "headers": {}
                },
                "env": {}
            }

        # Initialize MCP Manager with these configs
        self.mcp_manager = MCPManager(config=server_configs)
        print(f"Initialized MCP Manager with {len(server_configs)} server configurations")

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

    def generate_llm_test_arguments(self, tool_name: str, description: str,
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
            response = self.llm_client.chat.completions.create(
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

    def evaluate_response_with_llm(self, tool_name: str, description: str,
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
            response = self.llm_client.chat.completions.create(
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
        Connect to an MCP server using MCPManager.

        Args:
            server_name: Qualified name of the server (e.g., 'exa', '@user/server')

        Returns:
            MCPClient if successful, None otherwise
        """
        try:
            # Build client using MCP Manager with timeout
            client = await asyncio.wait_for(
                self.mcp_manager.build_client(server_name, timeout=20),
                timeout=25.0  # Slightly longer than internal timeout
            )
            return client

        except asyncio.TimeoutError:
            print(f"  [ERROR] Connection timeout (25s)")
            return None
        except asyncio.CancelledError:
            print(f"  [ERROR] Connection cancelled - likely auth error")
            return None
        except Exception as e:
            # Extract useful error message
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                print(f"  [ERROR] 401 Unauthorized - invalid API key for this server")
            elif "404" in error_msg:
                print(f"  [ERROR] 404 Not Found - server may not exist")
            elif "timeout" in error_msg.lower():
                print(f"  [ERROR] Connection timeout")
            elif "Cancelled" in error_msg:
                print(f"  [ERROR] Connection cancelled (likely auth error)")
            else:
                # Truncate long error messages
                short_msg = error_msg.split('\n')[0][:100]
                print(f"  [ERROR] Failed to connect: {short_msg}")
            return None

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
        client = await self.connect_to_server(server_name)
        if client is None:
            result["status"] = "connection_failed"
            result["error"] = "Failed to connect to server"
            return result

        try:
            # Generate test arguments
            input_schema = tool.get("inputSchema", {})

            if self.use_llm:
                print(f"    Generating LLM-based test arguments...")
                test_args = self.generate_llm_test_arguments(tool_name, tool_description, input_schema)
            else:
                test_args = self.generate_simple_test_arguments(input_schema)
                print(f"    Arguments: {json.dumps(test_args)}")

            result["test_arguments"] = test_args

            # Execute tool with timeout
            try:
                tool_result = await asyncio.wait_for(
                    client.execute_tool(tool_name, test_args),
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
                    evaluation = self.evaluate_response_with_llm(
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
            # Cleanup client
            try:
                await client.cleanup()
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
                       limit: Optional[int] = None):
        """
        Run tests on all tools.

        Args:
            server_filter: If provided, only test this server
            tool_filter: If provided, only test this tool
            limit: If provided, only test first N servers
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


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all MCP tools from tool_descriptions.ndjson",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--server", help="Test only this server (qualified name)")
    parser.add_argument("--tool", help="Test only this tool (requires --server)")
    parser.add_argument("--limit", type=int, help="Test only first N servers")
    parser.add_argument("--output", default="tool_test_results.json",
                       help="Output file for results (default: tool_test_results.json)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM to generate realistic test arguments")
    parser.add_argument("--evaluate-with-llm", action="store_true",
                       help="Use LLM to evaluate response quality (implies --use-llm)")
    parser.add_argument("--model", help="LLM model for argument generation (default: from OPENAI_MODEL env)")
    parser.add_argument("--judge-model", help="Judge model for evaluation (default: from JUDGE_MODEL env)")

    args = parser.parse_args()

    # Validate arguments
    if args.tool and not args.server:
        parser.error("--tool requires --server to be specified")

    # Set up paths (script is now in MCP_INFO_MGR directory)
    data_dir = PROJECT_ROOT / "mcp_data"
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
            limit=args.limit
        )

        # Save results
        tester.save_results(args.output)

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nFor LLM mode, ensure you have set one of:")
        print("  - OPENAI_API_KEY and optionally OPENAI_BASE_URL")
        print("  - OPENROUTER_API_KEY and optionally OPENROUTER_BASE_URL")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
