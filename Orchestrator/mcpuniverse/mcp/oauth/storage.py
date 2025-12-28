"""
OAuth token storage implementation for Smithery MCP servers.

Provides file-based persistent storage for OAuth tokens and client information.
Supports per-server token storage to allow multiple servers to maintain their own tokens.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


def _server_key(server_url: str) -> str:
    """Generate a safe filename key from server URL."""
    # Extract server name from URL (e.g., "reddit" from "https://server.smithery.ai/reddit")
    parsed = urlparse(server_url)
    path = parsed.path.strip('/')
    if path:
        # Use the path as the key (e.g., "reddit", "@user/server")
        # Replace / with _ for safe filenames
        return path.replace('/', '_').replace('@', '')
    # Fallback to hash of full URL
    return hashlib.sha256(server_url.encode()).hexdigest()[:16]


class FileTokenStorage:
    """
    File-based token storage for OAuth authentication.

    Stores tokens and client information in JSON files for persistence across sessions.
    Supports per-server storage to maintain separate tokens for each MCP server.
    """

    def __init__(self, storage_dir: str = None, server_url: str = None):
        """
        Initialize file-based token storage.

        Args:
            storage_dir: Directory to store token files.
                        Defaults to ~/.mcp/smithery_tokens/
            server_url: Server URL for per-server token storage.
                       If provided, tokens are stored in a server-specific file.
        """
        if storage_dir is None:
            # Default to project's working directory for easier sharing
            default_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                "MCP_INFO_MGR", "mcp_data", "working", "smithery_tokens"
            )
            # Fallback to home directory if project path doesn't exist
            if os.path.exists(os.path.dirname(default_dir)):
                storage_dir = default_dir
            else:
                storage_dir = os.path.expanduser("~/.mcp/smithery_tokens")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.server_url = server_url

        # Use per-server files if server_url is provided
        if server_url:
            key = _server_key(server_url)
            self.tokens_file = self.storage_dir / f"tokens_{key}.json"
            self.client_info_file = self.storage_dir / f"client_info_{key}.json"
        else:
            # Legacy single-file storage for backwards compatibility
            self.tokens_file = self.storage_dir / "tokens.json"
            self.client_info_file = self.storage_dir / "client_info.json"

    async def get_tokens(self) -> Optional[OAuthToken]:
        """Get stored OAuth tokens."""
        if not self.tokens_file.exists():
            return None

        try:
            with open(self.tokens_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return OAuthToken.model_validate(data)
        except (json.JSONDecodeError, Exception):
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store OAuth tokens."""
        with open(self.tokens_file, 'w', encoding='utf-8') as f:
            json.dump(tokens.model_dump(mode='json'), f, indent=2)

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        """Get stored client information."""
        if not self.client_info_file.exists():
            return None

        try:
            with open(self.client_info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return OAuthClientInformationFull.model_validate(data)
        except (json.JSONDecodeError, Exception):
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        with open(self.client_info_file, 'w', encoding='utf-8') as f:
            json.dump(client_info.model_dump(mode='json'), f, indent=2)

    def clear(self) -> None:
        """Clear all stored tokens and client info."""
        if self.tokens_file.exists():
            self.tokens_file.unlink()
        if self.client_info_file.exists():
            self.client_info_file.unlink()
