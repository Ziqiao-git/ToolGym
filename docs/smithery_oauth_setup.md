# Smithery Server OAuth Authentication Guide

## Overview

Smithery MCP servers now require **OAuth 2.0 authentication** instead of API Key authentication. This document explains how our system handles Smithery server connections with automatic OAuth authentication.

## Key Changes

### Previous Approach (API Key - No Longer Works)
```json
{
  "exa": {
    "streamable_http": {
      "url": "https://server.smithery.ai/exa/mcp?api_key={{SMITHERY_API_KEY}}"
    }
  }
}
```

❌ **Problem**: Smithery servers now reject API key authentication with `401 Unauthorized` errors.

### Current Approach (OAuth 2.0)
```json
{
  "exa": {
    "streamable_http": {
      "url": "https://server.smithery.ai/exa/mcp"
    }
  }
}
```

✅ **Solution**: OAuth 2.0 with PKCE (Proof Key for Code Exchange) flow, handled automatically by the system.

## How It Works

### 1. Automatic Smithery Detection

When Dynamic React Agent loads a server on-demand, it automatically detects Smithery servers:

```python
# In DynamicReActAgent._load_server_on_demand()
server_url = config.get("streamable_http", {}).get("url", "")
is_smithery = "smithery.ai" in server_url
```

### 2. OAuth Flow Initiation

For Smithery servers, the system automatically creates an OAuth provider:

```python
if is_smithery:
    from mcpuniverse.mcp.oauth import create_smithery_auth

    # Remove any API key from URL
    base_url = server_url.split("?")[0]

    auth_provider, callback_handler = create_smithery_auth(
        server_url=base_url,
        client_name=f"MCP Universe - {server_name}",
        redirect_port=8765,
        timeout=600.0
    )
```

### 3. OAuth Callback Handler

The callback handler starts a local HTTP server to receive OAuth redirect:

```python
if callback_handler:
    await callback_handler.__aenter__()
```

### 4. Client Connection with OAuth

The client connects to Smithery using the OAuth provider:

```python
client = await self._mcp_manager.build_client(
    server_name,
    transport="streamable_http",
    auth=auth_provider  # OAuth provider passed here
)
```

### 5. Token Storage and Reuse

OAuth tokens are stored in `~/.mcp/smithery_tokens/`:
- `tokens.json` - Contains access token and refresh token
- `client_info.json` - OAuth client registration info

**First connection**: Browser opens for user to authorize
**Subsequent connections**: Tokens are automatically reused and refreshed

## Architecture Changes

### MCPClient (client.py)

Added OAuth support to `connect_to_sse_server`:

```python
async def connect_to_sse_server(
    self,
    server_url: str,
    timeout: int = 20,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Any] = None  # NEW: OAuth support
):
    # ...
    transport = await self._exit_stack.enter_async_context(
        streamablehttp_client(server_url, headers=headers, auth=auth)
    )
```

### MCPManager (manager.py)

Added OAuth parameter to `build_client`:

```python
async def build_client(
    self,
    server_name: str,
    transport: str = "stdio",
    timeout: int = 30,
    mcp_gateway_address: str = "",
    auth: Optional[Any] = None  # NEW: OAuth support
) -> MCPClient:
    # ...
    await client.connect_to_sse_server(
        server_url=http_config.url,
        timeout=timeout,
        headers=http_config.headers or None,
        auth=auth  # Pass OAuth provider
    )
```

### DynamicReActAgent (dynamic_react.py)

Added automatic Smithery detection and OAuth handling in `_load_server_on_demand`:

```python
# Check if this is a Smithery server
server_url = config.get("streamable_http", {}).get("url", "")
is_smithery = "smithery.ai" in server_url

auth_provider = None
callback_handler = None

if is_smithery:
    # Create OAuth provider
    auth_provider, callback_handler = create_smithery_auth(...)
    await callback_handler.__aenter__()

try:
    # Connect with OAuth
    client = await self._mcp_manager.build_client(
        server_name,
        transport="streamable_http",
        auth=auth_provider
    )
finally:
    # Cleanup callback handler
    if callback_handler:
        await callback_handler.__aexit__(None, None, None)
```

## User Experience

### First Time Connection

When connecting to a Smithery server for the first time:

1. 🌐 **Browser opens automatically** to Smithery OAuth page
2. 👤 **User authorizes** the application
3. ✅ **Browser shows**: "Authentication successful! You can close this window."
4. 🔄 **Terminal continues** with server connection
5. 💾 **Tokens are saved** to `~/.mcp/smithery_tokens/`

### Subsequent Connections

All future connections:
- ⚡ **Automatic** - No browser interaction needed
- 🔑 **Reuse tokens** - Uses cached OAuth tokens
- 🔄 **Auto-refresh** - Tokens refreshed when expired

## Configuration Files

### remote_server_configs.json

All 116 Smithery servers updated to remove API key parameters:

```json
{
  "exa": {
    "streamable_http": {
      "url": "https://server.smithery.ai/exa/mcp"
    }
  },
  "@smithery-ai/github": {
    "streamable_http": {
      "url": "https://server.smithery.ai/smithery-ai/github/mcp"
    }
  },
  // ... 114 more servers
}
```

### No Environment Variables Required

Unlike the old API key approach, no `SMITHERY_API_KEY` environment variable is needed. OAuth tokens are managed automatically.

## Testing

### Test Smithery OAuth Standalone

```bash
python test_smithery_oauth_interactive.py
```

This tests OAuth authentication directly with Smithery servers.

### Test Dynamic React Agent with Smithery

```bash
python test_dynamic_react_smithery.py
```

This tests the full Dynamic React Agent flow with automatic Smithery OAuth.

### Test with Real Query

```bash
python runtime/run_react_agent.py "Search for machine learning repositories on GitHub"
```

This will:
1. Search for GitHub tools using meta-mcp
2. Find `@smithery-ai/github` server
3. **Automatically trigger OAuth** (browser opens on first run)
4. Load the GitHub server with OAuth
5. Execute the search

## Critical Implementation Details

### ClientSession Context Manager

**IMPORTANT**: Always use `ClientSession` as async context manager:

```python
# ✅ CORRECT:
async with ClientSession(read, write) as session:
    await session.initialize()
    # ... use session

# ❌ WRONG (will hang):
session = ClientSession(read, write)
await session.initialize()  # Hangs forever
```

**Why**: The `__aenter__` method starts the `_receive_loop` task that reads responses from the server. Without it, the session never receives any data.

### OAuth Callback Handler Lifecycle

The OAuth callback handler must be properly started and cleaned up:

```python
if callback_handler:
    await callback_handler.__aenter__()

try:
    # Connect to server
    client = await self._mcp_manager.build_client(...)
finally:
    # Always cleanup callback handler
    if callback_handler:
        await callback_handler.__aexit__(None, None, None)
```

## Troubleshooting

### Browser doesn't open for OAuth

**Cause**: OAuth callback handler not started or port conflict

**Solution**:
- Check that port 8765 is not in use
- Verify callback handler is properly initialized

### "401 Unauthorized" errors

**Cause**: Still using API key authentication or expired tokens

**Solution**:
- Verify server URL doesn't contain `?api_key=` parameter
- Delete cached tokens in `~/.mcp/smithery_tokens/` and re-authenticate
- Check that `is_smithery` detection is working

### Session hangs after OAuth

**Cause**: Not using `ClientSession` as async context manager

**Solution**: Use `async with ClientSession(...) as session:`

## Summary

Our system now provides **seamless OAuth authentication** for all Smithery servers:

✅ **Automatic detection** - Identifies Smithery servers automatically
✅ **OAuth flow** - Handles OAuth 2.0 with PKCE automatically
✅ **Token management** - Stores and refreshes tokens automatically
✅ **Minimal user interaction** - Browser opens only on first connection
✅ **Backward compatible** - Non-Smithery servers work as before

The implementation required minimal changes to 4 files while maintaining clean architecture and user experience.
