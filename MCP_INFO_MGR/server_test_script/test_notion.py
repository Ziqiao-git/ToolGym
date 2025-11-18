from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Construct server URL with authentication
from urllib.parse import urlencode
base_url = "https://server.smithery.ai/notion/mcp"
params = {"api_key": "eef36baa-b63a-47a9-b55f-0c0b4ea3dca0", "profile": "careful-gecko-T7eLN5"}
url = f"{base_url}?{urlencode(params)}"

async def main():
    # Connect to the server using HTTP client
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools_result = await session.list_tools()
            print(f"Available tools: {', '.join([t.name for t in tools_result.tools])}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())