"""
OAuth callback handler for Smithery authentication.

Provides a local HTTP server to receive OAuth callbacks and handle the authorization flow.
"""
import asyncio
import webbrowser
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading


class OAuthCallbackHandler:
    """
    Handles OAuth callbacks by running a local HTTP server.

    Opens a browser for user authorization and waits for the callback with the auth code.
    """

    def __init__(self, redirect_port: int = 8765):
        """
        Initialize OAuth callback handler.

        Args:
            redirect_port: Port for the local callback server. Defaults to 8765.
        """
        self.redirect_port = redirect_port
        self.redirect_uri = f"http://localhost:{redirect_port}/oauth/callback"

        # Callback data
        self._auth_code: Optional[str] = None
        self._state: Optional[str] = None
        self._error: Optional[str] = None
        self._received_event = asyncio.Event()

        # HTTP server
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop = None

    def reset_state(self):
        """
        Reset callback state for a new OAuth flow.

        MUST be called before each new OAuth flow when reusing the callback handler,
        otherwise the previous callback's state will cause a mismatch.
        """
        self._auth_code = None
        self._state = None
        self._error = None
        self._received_event = asyncio.Event()


    def _create_request_handler(self):
        """Create a request handler class with access to this instance."""
        callback_handler = self

        class CallbackRequestHandler(BaseHTTPRequestHandler):
            """HTTP request handler for OAuth callbacks."""

            def log_message(self, format, *args):
                """Suppress default logging."""
                pass

            def do_GET(self):
                """Handle GET request for OAuth callback."""
                parsed_path = urlparse(self.path)

                if parsed_path.path == '/oauth/callback':
                    query_params = parse_qs(parsed_path.query)

                    # Extract auth code and state
                    callback_handler._auth_code = query_params.get('code', [None])[0]
                    callback_handler._state = query_params.get('state', [None])[0]
                    callback_handler._error = query_params.get('error', [None])[0]

                    # Send response to browser
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

                    if callback_handler._error:
                        html = f"""
                        <html>
                        <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 50px;">
                            <h1 style="color: red;">❌ Authentication Failed</h1>
                            <p>Error: {callback_handler._error}</p>
                            <p>You can close this window.</p>
                        </body>
                        </html>
                        """
                    else:
                        html = """
                        <html>
                        <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 50px;">
                            <h1 style="color: green;">✓ Authentication Successful</h1>
                            <p>You can close this window and return to your application.</p>
                        </body>
                        </html>
                        """

                    self.wfile.write(html.encode())

                    # Signal that callback was received
                    if callback_handler._loop is not None:
                        callback_handler._loop.call_soon_threadsafe(
                            callback_handler._received_event.set
                        )
                else:
                    self.send_response(404)
                    self.end_headers()

        return CallbackRequestHandler

    def start_server(self):
        """Start the local HTTP server in a background thread."""
        handler_class = self._create_request_handler()

        # Smithery OAuth only accepts localhost:8765 as registered redirect URI
        # Do NOT fallback to other ports - it will cause "Unregistered redirect_uri" errors
        try:
            self._server = HTTPServer(('localhost', self.redirect_port), handler_class)
        except OSError as e:
            # Check for "Address already in use" error
            # errno 48 on macOS, 98 on Linux, 10048 on Windows
            if e.errno in (48, 98, 10048) or 'Address already in use' in str(e):
                raise RuntimeError(
                    f"Port {self.redirect_port} is already in use. "
                    f"Smithery OAuth requires exactly this port. "
                    f"Run 'lsof -i :{self.redirect_port}' to find the process, then kill it."
                ) from e
            else:
                raise

        def run_server():
            self._server.serve_forever()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

    def stop_server(self):
        """Stop the local HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=1)
            self._server_thread = None

    async def redirect_handler(self, authorization_url: str):
        """
        Handle authorization redirect by opening browser.

        Args:
            authorization_url: The OAuth authorization URL to open.
        """
        print(f"\n{'='*70}")
        print("🔐 Smithery OAuth Authorization Required")
        print(f"{'='*70}")
        print(f"Opening browser for authorization...\n")
        print(f"If the browser doesn't open automatically, visit this URL:")
        print(f"{authorization_url}\n")
        print(f"{'='*70}\n")

        # Open browser
        webbrowser.open(authorization_url)

    async def callback_handler(self) -> Tuple[str, Optional[str]]:
        """
        Wait for OAuth callback and return auth code and state.

        Returns:
            Tuple of (auth_code, state)

        Raises:
            RuntimeError: If authorization fails or times out.
        """
        # Wait for callback (with timeout)
        try:
            await asyncio.wait_for(self._received_event.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            raise RuntimeError("OAuth authorization timed out after 5 minutes")

        if self._error:
            raise RuntimeError(f"OAuth authorization failed: {self._error}")

        if not self._auth_code:
            raise RuntimeError("No authorization code received")

        return self._auth_code, self._state

    async def __aenter__(self):
        """Async context manager entry."""
        self._loop = asyncio.get_running_loop()
        # Only start server if not already running (may have been started early for port detection)
        if self._server is None:
            self.start_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.stop_server()
