"""
Quickly probe remote Smithery MCP endpoints by attempting a lightweight MCP handshake.

This script scans an NDJSON metadata dump (e.g. from fetch_metadata.py),
filters servers where `remote` is true, and for each connection entry attempts
to establish a streamable HTTP MCP session and list available tools. This
provides a stronger signal than a plain HTTP GET probe.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


ENV_VAR_API_KEY = "SMITHERY_API_KEY"
DEFAULT_TIMEOUT = 10
DEFAULT_SLEEP = 0.2


@dataclass
class ReachabilityResult:
    qualified_name: str
    url: str
    status: str
    tool_count: Optional[int] = None
    error: Optional[str] = None


def iter_ndjson(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def load_local_env(env_filename: str = ".env") -> None:
    """
    Populate os.environ with key/value pairs from a sibling .env file.

    Only keys that are not already present in the environment are set.
    """
    env_path = Path(__file__).resolve().with_name(env_filename)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


async def _probe_endpoint(url: str, timeout: int, headers: Optional[dict[str, str]] = None) -> ReachabilityResult:
    async with streamablehttp_client(
        url=url,
        headers=headers,
        sse_read_timeout=timedelta(seconds=timeout),
    ) as (read_stream, write_stream, _):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=timeout),
        ) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tool_list = getattr(tools_response, "tools", None)
            tool_count = len(tool_list) if tool_list is not None else None
            return ReachabilityResult(
                qualified_name="",
                url=url,
                status="ok",
                tool_count=tool_count,
            )


async def probe_endpoint(url: str, timeout: int, headers: Optional[dict[str, str]] = None) -> ReachabilityResult:
    try:
        return await asyncio.wait_for(_probe_endpoint(url, timeout, headers=headers), timeout=timeout)
    except asyncio.TimeoutError:
        return ReachabilityResult(
            qualified_name="",
            url=url,
            status="error: Timeout",
            error=f"Timed out after {timeout}s",
        )
    except asyncio.CancelledError as exc:
        return ReachabilityResult(
            qualified_name="",
            url=url,
            status="error: Cancelled",
            error=str(exc),
        )
    except ExceptionGroup as exc_group:  # type: ignore[name-defined]
        sub_messages = "; ".join(f"{type(err).__name__}: {err}" for err in exc_group.exceptions)
        # Try to detect HTTP status codes embedded in the exception chain
        status_label = "error: ExceptionGroup"
        for err in exc_group.exceptions:
            response = getattr(err, "response", None)
            if response is not None:
                status_code = getattr(response, "status", None) or getattr(response, "status_code", None)
                if status_code:
                    status_label = f"http-{status_code}"
                    break
        return ReachabilityResult(
            qualified_name="",
            url=url,
            status=status_label,
            error=sub_messages or str(exc_group),
        )
    except Exception as exc:
        status_label = f"error: {exc.__class__.__name__}"
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status", None) or getattr(response, "status_code", None)
            if status_code:
                status_label = f"http-{status_code}"
        return ReachabilityResult(
            qualified_name="",
            url=url,
            status=status_label,
            error=str(exc),
        )


def add_query_params(url: str, params: dict[str, str]) -> str:
    """
    Return a copy of `url` with provided query parameters added if missing.

    Existing parameters take precedence.
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        if not value:
            continue
        query.setdefault(key, value)
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def redact_api_key(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "api_key" in query:
        query["api_key"] = "***"
    if "apiKey" in query:
        query["apiKey"] = "***"
    redacted_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=redacted_query))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check reachability of remote Smithery MCP endpoints.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("smithery_metadata.ndjson"),
        help="NDJSON metadata file (default: smithery_metadata.ndjson).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Sleep between requests in seconds (default: {DEFAULT_SLEEP}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of servers to probe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file to append reachability results.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Smithery API key (overrides environment/.env).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Smithery profile to append as query parameter when required.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    if not args.input.exists():
        print(f"Error: metadata file {args.input} not found.", file=sys.stderr)
        return 1

    api_key = args.api_key or os.getenv(ENV_VAR_API_KEY)
    if not api_key:
        print(f"Error: set {ENV_VAR_API_KEY} in your environment or .env file.", file=sys.stderr)
        return 1
    profile = args.profile or os.getenv("SMITHERY_PROFILE")
    headers = {"Authorization": f"Bearer {api_key}"}

    processed = 0
    status_counts: dict[str, int] = {}

    output_handle = args.output.open("a", encoding="utf-8") if args.output else None

    try:
        for record in iter_ndjson(args.input):
            # Support two formats:
            # 1. Original format: {remote: true, qualifiedName: "...", connections: [...]}
            # 2. Simple format: {qualifiedName: "...", requestedUrl: "..."}

            urls_to_probe = []
            qualified_name = record.get("qualifiedName", "<unknown>")

            # Format 1: Check for original format with remote and connections
            if record.get("remote") is True:
                connections = record.get("connections") or []
                for connection in connections:
                    url = connection.get("deploymentUrl") or connection.get("url")
                    if url:
                        urls_to_probe.append(url)

            # Format 2: Check for simple format with requestedUrl
            elif "requestedUrl" in record:
                url = record.get("requestedUrl")
                if url:
                    # Remove existing api_key parameter if present (will be re-added)
                    parsed = urlparse(url)
                    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
                    # Remove redacted keys
                    query.pop("api_key", None)
                    query.pop("apiKey", None)
                    clean_query = urlencode(query, doseq=True)
                    clean_url = urlunparse(parsed._replace(query=clean_query))
                    urls_to_probe.append(clean_url)

            # Skip if no URLs found
            if not urls_to_probe:
                continue

            for url in urls_to_probe:
                auth_url = add_query_params(
                    url,
                    {
                        "api_key": api_key,
                        "profile": profile,
                    },
                )

                result = asyncio.run(probe_endpoint(auth_url, args.timeout, headers=headers))
                if result is None:
                    result = ReachabilityResult(
                        qualified_name="",
                        url=url,
                        status="error: Unknown",
                        error="Probe returned no result",
                    )
                result.qualified_name = qualified_name
                result.url = url

                processed += 1
                label = result.status
                status_counts[label] = status_counts.get(label, 0) + 1

                tool_info = f"tools={result.tool_count}" if result.tool_count is not None else ""
                error_info = f"error={result.error}" if result.error and label != "ok" else ""
                info = " ".join(part for part in [tool_info, error_info] if part)
                info_suffix = f" {info}" if info else ""
                print(f"[{label:22}] {qualified_name:40} {redact_api_key(auth_url)}{info_suffix}")

                if output_handle:
                    payload = {
                        "qualifiedName": result.qualified_name,
                        "url": result.url,
                        "status": result.status,
                        "toolCount": result.tool_count,
                        "error": result.error,
                        "requestedUrl": redact_api_key(auth_url),
                    }
                    output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

                if args.sleep > 0:
                    time.sleep(args.sleep)

            if args.limit is not None and processed >= args.limit:
                break
    finally:
        if output_handle:
            output_handle.close()

    print("\nSummary:")
    print(f"  Total probes: {processed}")
    for label, count in sorted(status_counts.items(), key=lambda item: item[0]):
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
