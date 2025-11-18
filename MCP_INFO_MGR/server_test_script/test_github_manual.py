#!/usr/bin/env python3
"""
Manual test script for @smithery-ai/github server with LLM evaluation
Uses direct MCP ClientSession connection and evaluates tools using LLM
"""
import asyncio
import json
import sys
import os
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import datetime

# Import MCP client directly
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Import OpenAI for LLM evaluation
from openai import OpenAI

load_dotenv()


async def test_and_evaluate_github_server():
    """Test the @smithery-ai/github server and evaluate using LLM."""

    print("=" * 80)
    print("Testing @smithery-ai/github Server with LLM Evaluation")
    print("=" * 80)

    # Get authentication from environment
    smithery_key = os.getenv('SMITHERY_API_KEY', '')
    smithery_profile = os.getenv('SMITHERY_PROFILE', '')
    openrouter_key = os.getenv('OPENROUTER_API_KEY', '')
    openrouter_base = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

    if not smithery_key:
        print("\n❌ ERROR: SMITHERY_API_KEY not found in environment")
        print("   Set SMITHERY_API_KEY in your .env file\n")
        return False

    if not smithery_profile:
        print("\n❌ ERROR: SMITHERY_PROFILE not found in environment")
        print("   You need to set SMITHERY_PROFILE (e.g., 'careful-gecko-T7eLN5')")
        return False

    if not openrouter_key:
        print("\n❌ ERROR: OPENROUTER_API_KEY not found in environment")
        print("   LLM evaluation requires OpenRouter API key")
        return False

    # Initialize OpenAI client for LLM evaluation
    llm_client = OpenAI(
        api_key=openrouter_key,
        base_url=openrouter_base
    )

    # Construct server URL with authentication
    base_url = "https://server.smithery.ai/@smithery-ai/github/mcp"
    params = {
        "api_key": smithery_key,
        "profile": smithery_profile
    }
    url = f"{base_url}?{urlencode(params)}"

    print(f"\n1. Connecting to GitHub MCP server...")
    print(f"   URL: {base_url}")
    print(f"   Profile: {smithery_profile}")

    try:
        # Connect to the server using HTTP client
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                print("\n2. Initializing session...")
                await session.initialize()

                print("3. Listing available tools...")
                tools_result = await session.list_tools()
                tools = tools_result.tools

                print(f"\n✅ Connection successful! Found {len(tools)} tools")

                # Prepare tool data for LLM evaluation
                tools_data = []
                for tool in tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description if hasattr(tool, 'description') else 'No description',
                        "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    }
                    tools_data.append(tool_info)

                print("\n4. Testing sample tool execution...")
                # Test search_repositories
                test_results = {}
                if any(t.name == "search_repositories" for t in tools):
                    print("   Executing: search_repositories(query='mcp-server')")
                    try:
                        result = await session.call_tool(
                            "search_repositories",
                            arguments={"query": "mcp-server"}
                        )
                        test_results["search_repositories"] = {
                            "status": "success",
                            "result_preview": str(result)[:500]
                        }
                        print(f"   ✅ Success: Retrieved repository search results")
                    except Exception as e:
                        test_results["search_repositories"] = {
                            "status": "error",
                            "error": str(e)
                        }
                        print(f"   ❌ Error: {e}")

                # Test get_repository
                if any(t.name == "get_repository" for t in tools):
                    print("   Executing: get_repository(owner='modelcontextprotocol', repo='servers')")
                    try:
                        result = await session.call_tool(
                            "get_repository",
                            arguments={"owner": "modelcontextprotocol", "repo": "servers"}
                        )
                        test_results["get_repository"] = {
                            "status": "success",
                            "result_preview": str(result)[:500]
                        }
                        print(f"   ✅ Success: Retrieved repository details")
                    except Exception as e:
                        test_results["get_repository"] = {
                            "status": "error",
                            "error": str(e)
                        }
                        print(f"   ❌ Error: {e}")

                print("\n5. Evaluating tools using LLM (gpt-4o-mini)...")

                # Prepare evaluation prompt
                evaluation_prompt = f"""You are evaluating a GitHub MCP server that provides {len(tools)} tools for GitHub API operations.

**Available Tools:**
{json.dumps(tools_data, indent=2)}

**Sample Execution Results:**
{json.dumps(test_results, indent=2)}

Please evaluate this MCP server on the following dimensions (score 0-10 for each):

1. **Tool Coverage** (0-10): How comprehensive is the coverage of GitHub API functionality?
   - Consider: repositories, issues, PRs, code search, file operations, branches, commits, etc.

2. **Tool Quality** (0-10): How well-designed are the tool schemas and descriptions?
   - Consider: clarity of descriptions, appropriate input parameters, proper data types

3. **Practical Utility** (0-10): How useful would these tools be for real-world GitHub automation tasks?
   - Consider: common use cases, workflow automation, CI/CD integration

4. **Execution Reliability** (0-10): Based on the sample executions, how reliable is the server?
   - Consider: success rate, error handling, response quality

5. **API Completeness** (0-10): How complete is the GitHub API coverage compared to what's possible?
   - Consider: missing critical features, depth of functionality

Provide your evaluation in the following JSON format:
{{
  "tool_coverage": <score>,
  "tool_quality": <score>,
  "practical_utility": <score>,
  "execution_reliability": <score>,
  "api_completeness": <score>,
  "overall_score": <average of 5 scores>,
  "strengths": ["list", "of", "key", "strengths"],
  "weaknesses": ["list", "of", "areas", "for", "improvement"],
  "recommended_use_cases": ["list", "of", "ideal", "use", "cases"],
  "summary": "2-3 sentence overall assessment"
}}

Respond ONLY with valid JSON, no other text."""

                # Call LLM for evaluation
                response = llm_client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of API tools and developer tooling. You provide structured, objective assessments."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )

                evaluation_text = response.choices[0].message.content.strip()

                # Parse evaluation JSON
                # Remove markdown code blocks if present
                if evaluation_text.startswith("```json"):
                    evaluation_text = evaluation_text[7:]
                if evaluation_text.startswith("```"):
                    evaluation_text = evaluation_text[3:]
                if evaluation_text.endswith("```"):
                    evaluation_text = evaluation_text[:-3]

                evaluation = json.loads(evaluation_text.strip())

                # Display evaluation results
                print("\n" + "=" * 80)
                print("LLM EVALUATION RESULTS")
                print("=" * 80)
                print(f"\n📊 Scores (0-10 scale):")
                print(f"   Tool Coverage:        {evaluation['tool_coverage']}/10")
                print(f"   Tool Quality:         {evaluation['tool_quality']}/10")
                print(f"   Practical Utility:    {evaluation['practical_utility']}/10")
                print(f"   Execution Reliability: {evaluation['execution_reliability']}/10")
                print(f"   API Completeness:     {evaluation['api_completeness']}/10")
                print(f"\n   ⭐ Overall Score:      {evaluation['overall_score']:.1f}/10")

                print(f"\n💪 Strengths:")
                for strength in evaluation['strengths']:
                    print(f"   • {strength}")

                print(f"\n⚠️  Areas for Improvement:")
                for weakness in evaluation['weaknesses']:
                    print(f"   • {weakness}")

                print(f"\n🎯 Recommended Use Cases:")
                for use_case in evaluation['recommended_use_cases']:
                    print(f"   • {use_case}")

                print(f"\n📝 Summary:")
                print(f"   {evaluation['summary']}")

                # Save full evaluation to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"mcp_data/reports/github_server_evaluation_{timestamp}.json"

                full_report = {
                    "server_name": "@smithery-ai/github",
                    "timestamp": timestamp,
                    "tools_count": len(tools),
                    "tools": tools_data,
                    "test_results": test_results,
                    "evaluation": evaluation
                }

                os.makedirs("mcp_data/reports", exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(full_report, f, indent=2)

                print(f"\n💾 Full evaluation saved to: {output_file}")

                print("\n" + "=" * 80)
                print("Test and evaluation completed successfully!")
                print("=" * 80)
                return True

    except asyncio.TimeoutError as e:
        print(f"\n❌ TIMEOUT: Server took too long to respond")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   Message: {e}")

        import traceback
        print(f"\n   Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_and_evaluate_github_server())
    sys.exit(0 if success else 1)
