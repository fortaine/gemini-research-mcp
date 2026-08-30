"""Tests for dual transport support (stdio default, opt-in streamable-http).

Covers:
- loopback-host detection and the "refuse to start" safety guard
- CLI/env var driven transport configuration in main()
- a live streamable-http smoke test proving sessionless (no sticky session)
  behavior across independent calls
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import patch

import pytest

from gemini_research_mcp.server import (
    _is_loopback_host,
    _require_auth_for_non_loopback,
    main,
    mcp,
)


class TestLoopbackDetection:
    """`_is_loopback_host` classifies hosts as local-only or not."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_hosts(self, host):
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "192.168.1.10", "example.com", "10.0.0.5"],
    )
    def test_non_loopback_hosts(self, host):
        """0.0.0.0/:: bind all interfaces and must NOT be treated as loopback."""
        assert _is_loopback_host(host) is False


class TestAuthGuard:
    """`_require_auth_for_non_loopback` fails closed without authentication."""

    def test_loopback_without_auth_is_allowed(self):
        _require_auth_for_non_loopback(host="127.0.0.1", has_auth=False)  # no raise

    def test_loopback_with_auth_is_allowed(self):
        _require_auth_for_non_loopback(host="localhost", has_auth=True)  # no raise

    def test_non_loopback_without_auth_refuses(self):
        with pytest.raises(SystemExit, match="Refusing to bind"):
            _require_auth_for_non_loopback(host="0.0.0.0", has_auth=False)

    def test_non_loopback_with_auth_is_allowed(self):
        _require_auth_for_non_loopback(host="0.0.0.0", has_auth=True)  # no raise


@pytest.fixture(autouse=True)
def _reset_mcp_auth():
    """Ensure `mcp.auth` mutations in one test never leak into another."""
    original = mcp.auth
    yield
    mcp.auth = original


class TestMainTransportConfig:
    """`main()` wires CLI/env args into transport selection without starting a real server."""

    def test_default_transport_is_stdio(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["gemini-research-mcp"])
        with patch.object(mcp, "run") as mock_run:
            main()
        mock_run.assert_called_once_with(transport="stdio")

    def test_explicit_streamable_http_transport(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gemini-research-mcp",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                "9123",
                "--path",
                "/custom-mcp",
            ],
        )
        with patch.object(mcp, "run") as mock_run:
            main()
        mock_run.assert_called_once_with(
            transport="streamable-http",
            host="127.0.0.1",
            port=9123,
            path="/custom-mcp",
            stateless_http=True,
        )

    def test_env_var_selects_streamable_http(self, monkeypatch):
        monkeypatch.setenv("GEMINI_RESEARCH_TRANSPORT", "streamable-http")
        monkeypatch.setenv("GEMINI_RESEARCH_HTTP_HOST", "127.0.0.1")
        monkeypatch.setattr(sys, "argv", ["gemini-research-mcp"])
        with patch.object(mcp, "run") as mock_run:
            main()
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["transport"] == "streamable-http"

    def test_non_loopback_without_bearer_token_refuses_to_start(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["gemini-research-mcp", "--transport", "streamable-http", "--host", "0.0.0.0"],
        )
        monkeypatch.delenv("GEMINI_RESEARCH_HTTP_BEARER_TOKEN", raising=False)
        with (
            patch.object(mcp, "run") as mock_run,
            pytest.raises(SystemExit, match="Refusing to bind"),
        ):
            main()
        mock_run.assert_not_called()

    def test_bearer_token_enables_non_loopback_binding(self, monkeypatch):
        monkeypatch.setenv("GEMINI_RESEARCH_HTTP_BEARER_TOKEN", "super-secret-token")
        monkeypatch.setattr(
            sys,
            "argv",
            ["gemini-research-mcp", "--transport", "streamable-http", "--host", "0.0.0.0"],
        )
        with patch.object(mcp, "run") as mock_run:
            main()
        mock_run.assert_called_once()
        assert mcp.auth is not None

    def test_bearer_token_is_never_gemini_api_key(self, monkeypatch):
        """GEMINI_API_KEY must never double as the MCP client bearer token."""
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-provider-secret")
        monkeypatch.delenv("GEMINI_RESEARCH_HTTP_BEARER_TOKEN", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["gemini-research-mcp", "--transport", "streamable-http", "--host", "0.0.0.0"],
        )
        with patch.object(mcp, "run"), pytest.raises(SystemExit):
            main()
        # Even though an API key is present, it must not have been used as auth.
        assert mcp.auth is None


class TestStreamableHttpLive:
    """A real streamable-http server round-trip, proving sessionless behavior."""

    @pytest.mark.asyncio
    async def test_sessionless_calls_across_independent_requests(self):
        from fastmcp import Client, FastMCP

        probe = FastMCP(name="dual-transport-probe")

        @probe.tool
        def echo(value: str) -> str:
            return value

        host, port, path = "127.0.0.1", 8934, "/mcp"
        server_task = asyncio.create_task(
            probe.run_async(
                transport="streamable-http",
                host=host,
                port=port,
                path=path,
                stateless_http=True,
                show_banner=False,
            )
        )
        try:
            await asyncio.sleep(0.75)
            url = f"http://{host}:{port}{path}"

            # First independent client/connection.
            async with Client(url) as client_a:
                tools = await client_a.list_tools()
                assert any(t.name == "echo" for t in tools)
                result_a = await client_a.call_tool("echo", {"value": "first"})
                assert result_a.data == "first"

            # A brand-new connection must work with no sticky session state
            # carried over from the first - this is the sessionless contract.
            async with Client(url) as client_b:
                result_b = await client_b.call_tool("echo", {"value": "second"})
                assert result_b.data == "second"
        finally:
            server_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server_task
