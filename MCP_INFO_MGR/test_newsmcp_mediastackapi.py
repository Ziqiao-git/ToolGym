"""
Minimal Smithery OAuth test against @rftsngl/newsmcp_mediastackapi.

Run:
  python MCP_INFO_MGR/server_test_script/test_newsmcp_mediastackapi.py

On first run a browser opens for Smithery OAuth; tokens are cached under ~/.mcp/smithery_tokens/.
"""
import asyncio
import sys
from pathlib import Path

# Make Orchestrator imports available without installing as a package.
# File now lives in MCP_INFO_MGR/, so repo root is parents[1].
REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_DIR = REPO_ROOT / "Orchestrator"
sys.path.insert(0, str(ORCH_DIR))

from mcp import ClientSession  # type: ignore
from mcp.client.streamable_http import streamablehttp_client  # type: ignore
from mcpuniverse.mcp.oauth import create_smithery_auth  # type: ignore


SERVER_URL = "https://server.smithery.ai/@rftsngl/newsmcp_mediastackapi/mcp"


async def main() -> None:
    auth_provider, callback_handler = create_smithery_auth(
        server_url=SERVER_URL,
        client_name="Local newsmcp test",
    )

    async with callback_handler:
        async with streamablehttp_client(url=SERVER_URL, auth=auth_provider) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tool_names = ", ".join(tool.name for tool in tools_result.tools)
                print(f"Tools: {tool_names}")

                # Smoke-test the main tool with a small query
                resp = await session.call_tool(
                    "get_latest_news",
                    {"keywords": "ai", "limit": 3},
                )
                print("get_latest_news response:")
                print(resp)


if __name__ == "__main__":
    asyncio.run(main())
