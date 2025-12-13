"""
Smithery-specific OAuth authentication helpers.

Provides convenient functions to create OAuth authentication for Smithery MCP servers.
"""
from typing import Optional

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata

from .storage import FileTokenStorage
from .callback_handler import OAuthCallbackHandler


def create_smithery_auth(
    server_url: str,
    client_name: str = "MCP Universe Client",
    storage_dir: Optional[str] = None,
    redirect_port: int = 8765,
    timeout: float = 300.0
) -> tuple[OAuthClientProvider, OAuthCallbackHandler]:
    """
    Create OAuth authentication provider for Smithery MCP servers.

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
    # Create token storage
    storage = FileTokenStorage(storage_dir=storage_dir)

    # Create callback handler
    callback_handler = OAuthCallbackHandler(redirect_port=redirect_port)

    # Create OAuth client metadata
    client_metadata = OAuthClientMetadata(
        client_name=client_name,
        client_uri="http://localhost",
        redirect_uris=[callback_handler.redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="read write",
        token_endpoint_auth_method="none"
    )

    # Create OAuth provider
    auth_provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=callback_handler.redirect_handler,
        callback_handler=callback_handler.callback_handler,
        timeout=timeout
    )

    return auth_provider, callback_handler
