"""
OAuth token storage implementation for Smithery MCP servers.

Provides file-based persistent storage for OAuth tokens and client information.
"""
import json
import os
from pathlib import Path
from typing import Optional

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


class FileTokenStorage:
    """
    File-based token storage for OAuth authentication.

    Stores tokens and client information in JSON files for persistence across sessions.
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize file-based token storage.

        Args:
            storage_dir: Directory to store token files.
                        Defaults to ~/.mcp/smithery_tokens/
        """
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.mcp/smithery_tokens")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

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
