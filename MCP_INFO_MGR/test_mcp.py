"""
Minimal Smithery OAuth test - lists tools and calls the first one.

Run:
  python MCP_INFO_MGR/test_mcp.py

On first run a browser opens for Smithery OAuth; tokens are cached under ~/.mcp/smithery_tokens/.
"""
import asyncio
import os
import sys
import json
from pathlib import Path

# Make Orchestrator imports available without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_DIR = REPO_ROOT / "Orchestrator"
sys.path.insert(0, str(ORCH_DIR))

from mcp import ClientSession  # type: ignore
from mcp.client.streamable_http import streamablehttp_client  # type: ignore
from mcpuniverse.mcp.oauth import create_smithery_auth  # type: ignore


SERVER_URL = "https://server.smithery.ai/asana"


async def main() -> None:
    # Allow overriding token cache location (e.g., persistent volume) via env.
    storage_dir = os.getenv("SMITHERY_TOKEN_DIR")

    auth_provider, callback_handler = create_smithery_auth(
        server_url=SERVER_URL,
        client_name="Local MCP test",
        storage_dir=storage_dir,
    )

    async with callback_handler:
        async with streamablehttp_client(url=SERVER_URL, auth=auth_provider) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List all tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"Found {len(tools)} tools:")
                for i, tool in enumerate(tools):
                    print(f"  [{i}] {tool.name}: {tool.description[:80] if tool.description else 'No description'}...")

                if not tools:
                    print("No tools available!")
                    return

                # Get the first tool
                first_tool = tools[0]
                print(f"\n--- Testing first tool: {first_tool.name} ---")
                print(f"Description: {first_tool.description}")
                print(f"Input schema: {json.dumps(first_tool.inputSchema, indent=2)}")

               

if __name__ == "__main__":
    asyncio.run(main())
