"""
Smithery-specific OAuth authentication helpers.

Provides convenient functions to create OAuth authentication for Smithery MCP servers.
"""
from typing import Optional

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata

from .storage import FileTokenStorage
from .callback_handler import OAuthCallbackHandler


# Singleton for shared callback handler (reused across servers to avoid port conflicts)
_shared_callback_handler = None
_shared_client_metadata = None
_storage_dir = None


def create_smithery_auth(
    server_url: str,
    client_name: str = "MCP Universe Client",
    storage_dir: Optional[str] = None,
    redirect_port: int = 8765,
    timeout: float = 300.0
) -> tuple[OAuthClientProvider, OAuthCallbackHandler]:
    """
    Create OAuth authentication provider for Smithery MCP servers.

    Uses per-server token storage so each server maintains its own OAuth tokens.
    The callback handler is shared to avoid port conflicts, but its state is reset
    before each new OAuth flow.

    Args:
        server_url: The Smithery server URL (e.g., https://server.smithery.ai/exa/mcp)
        client_name: Name of your application. Defaults to "MCP Universe Client".
        storage_dir: Directory to store tokens. Defaults to ~/.mcp/smithery_tokens/
        redirect_port: Port for OAuth callback. Defaults to 8765.
        timeout: OAuth flow timeout in seconds. Defaults to 300 (5 minutes).

    Returns:
        Tuple of (OAuthClientProvider, OAuthCallbackHandler)

    Example:
        >>> auth_provider, callback_handler = create_smithery_auth(
        ...     server_url="https://server.smithery.ai/exa/mcp",
        ...     client_name="My AI App"
        ... )
        >>> async with callback_handler:
        ...     transport = await streamablehttp_client(
        ...         url=server_url,
        ...         auth=auth_provider
        ...     )
    """
    global _shared_callback_handler, _shared_client_metadata, _storage_dir

    # Remember storage dir for consistency
    if _storage_dir is None:
        _storage_dir = storage_dir

    # Reuse shared callback handler to avoid port conflicts
    if _shared_callback_handler is None:
        _shared_callback_handler = OAuthCallbackHandler(redirect_port=redirect_port)
        # Start server immediately to determine actual port (may fallback if port is busy)
        # This ensures redirect_uri has the correct port before creating client metadata
        _shared_callback_handler.start_server()

    # Reuse shared client metadata (created AFTER server starts so we have correct port)
    if _shared_client_metadata is None:
        _shared_client_metadata = OAuthClientMetadata(
            client_name=client_name,
            client_uri="http://localhost",
            redirect_uris=[_shared_callback_handler.redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="read write",
            token_endpoint_auth_method="none"
        )

    # IMPORTANT: Reset callback handler state before each new OAuth flow
    # This prevents state mismatch when reusing the callback handler
    _shared_callback_handler.reset_state()

    # Create PER-SERVER storage so each server has its own tokens
    # This follows Smithery's requirement that tokens are server-specific
    server_storage = FileTokenStorage(storage_dir=_storage_dir, server_url=server_url)

    # Create a NEW auth provider for EACH server URL with its own storage
    auth_provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=_shared_client_metadata,
        storage=server_storage,
        redirect_handler=_shared_callback_handler.redirect_handler,
        callback_handler=_shared_callback_handler.callback_handler,
        timeout=timeout
    )

    return auth_provider, _shared_callback_handler


def reset_shared_auth():
    """Reset shared OAuth resources. Useful for testing or cleanup."""
    global _shared_callback_handler, _shared_client_metadata, _storage_dir
    _shared_callback_handler = None
    _shared_client_metadata = None
    _storage_dir = None
