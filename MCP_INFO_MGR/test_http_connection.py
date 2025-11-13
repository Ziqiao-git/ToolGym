#!/usr/bin/env python3
"""
Test raw HTTP connection to MCP server to see what the server returns.
"""
import asyncio
import httpx
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "Orchestrator"
load_dotenv(str(ORCHESTRATOR_DIR / ".env"))


async def test_http(server_name: str):
    """Test raw HTTP connection."""
    api_key = os.getenv("SMITHERY_API_KEY")
    url = f"https://server.smithery.ai/{server_name}/mcp"

    print("=" * 80)
    print(f"Testing HTTP connection: {server_name}")
    print("=" * 80)
    print(f"URL: {url}")
    print()

    # Test 1: Simple GET request
    print("Test 1: Simple GET request (check if server exists)")
    print("-" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                url,
                params={"api_key": api_key}
            )
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"\nBody (first 500 chars):")
            print(response.text[:500])
            print()

            if response.status_code == 200:
                print("✅ Server is reachable")
            elif response.status_code == 401:
                print("❌ Authentication failed (401)")
            elif response.status_code == 404:
                print("❌ Server not found (404)")
            else:
                print(f"⚠️  Unexpected status: {response.status_code}")

        except Exception as e:
            print(f"❌ Request failed: {e}")

    print("\n" + "=" * 80)

    # Test 2: MCP initialization POST
    print("Test 2: MCP initialize request (what the MCP client does)")
    print("-" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # This mimics what the MCP client sends
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }

            response = await client.post(
                url,
                params={"api_key": api_key},
                json=init_request,
                headers={"Content-Type": "application/json"}
            )

            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"\nBody:")
            print(response.text[:1000])

            if response.status_code == 200:
                print("\n✅ Server accepted initialize request")
            else:
                print(f"\n❌ Server returned {response.status_code}")

        except Exception as e:
            print(f"❌ Request failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_http_connection.py <server_name>")
        print("Example: python test_http_connection.py exa")
        print("Example: python test_http_connection.py @smithery-ai/github")
        sys.exit(1)

    asyncio.run(test_http(sys.argv[1]))
