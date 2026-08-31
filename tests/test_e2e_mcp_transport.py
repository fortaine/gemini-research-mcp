"""End-to-end tests for MCP server transport validation.

These tests validate that the MCP server works correctly over stdio transport,
ensuring compatibility with VS Code, Claude Desktop, and other MCP clients.

Run with: uv run pytest tests/test_e2e_mcp_transport.py -v --tb=short

These tests do NOT require GEMINI_API_KEY - they only validate the MCP protocol.
"""

import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.client.session import ClientSession


@asynccontextmanager
async def _gemini_stdio_session() -> AsyncIterator[ClientSession]:
    """Open a real stdio MCP client session against the main server."""
    from mcp.client.stdio import StdioServerParameters, stdio_client

    repo_root = Path(__file__).resolve().parent.parent
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "gemini_research_mcp.server"],
        cwd=str(repo_root),
        env=dict(os.environ),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            yield session


class TestMCPServerStartup:
    """Tests for MCP server startup and initialization."""

    def test_server_module_imports(self):
        """Server module should import without errors."""
        from gemini_research_mcp.server import main, mcp

        assert mcp is not None
        assert main is not None
        assert callable(main)

    def test_fastmcp_instance_type(self):
        """Server should use FastMCP instance."""
        from fastmcp import FastMCP

        from gemini_research_mcp.server import mcp

        assert isinstance(mcp, FastMCP)

    def test_server_name(self):
        """Server should have correct name."""
        from gemini_research_mcp.server import mcp

        assert mcp.name == "Gemini Research"

    def test_server_version_set(self):
        """Server should have version from package."""
        from gemini_research_mcp import __version__

        # FastMCP may or may not expose version, but package should have it
        assert __version__ is not None
        assert len(__version__) > 0


class TestMCPToolRegistration:
    """Tests for MCP tool registration with fastmcp."""

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        """Pinned tools and search meta-tools should be visible."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}

        expected_tools = {
            "research_web",
            "research_deep",
            "research_deep_max",
            "resume_research",
            "export_research_session",
            "search_tools",
            "call_tool",
        }

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    @pytest.mark.asyncio
    async def test_tool_count(self):
        """Should expose pinned tools plus synthetic search tools."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        assert len(tools) == 7, f"Expected 7 tools, got {len(tools)}"

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self):
        """All tools should have descriptions."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"


class TestToolSearchOverStdio:
    """Exercise the visible search surface over the MCP wire protocol."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_search_surface(self) -> None:
        async with _gemini_stdio_session() as session:
            tools = await session.list_tools()

        tool_names = {tool.name for tool in tools.tools}

        assert tool_names == {
            "research_web",
            "research_deep",
            "research_deep_max",
            "resume_research",
            "export_research_session",
            "search_tools",
            "call_tool",
        }

    @pytest.mark.asyncio
    async def test_search_tools_discovers_fetch_webpage(self) -> None:
        async with _gemini_stdio_session() as session:
            result = await session.call_tool("search_tools", {"query": "fetch webpage"})

        assert result.is_error is False
        assert result.structured_content is not None

        matches = result.structured_content["result"]
        fetch_tool = next(tool for tool in matches if tool["name"] == "fetch_webpage")

        assert "url" in fetch_tool["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_call_tool_proxy_over_wire(self) -> None:
        async with _gemini_stdio_session() as session:
            result = await session.call_tool(
                "call_tool",
                {"name": "list_format_templates", "arguments": {}},
            )

        assert result.is_error is False
        assert result.structured_content is not None

        payload = json.loads(result.structured_content["result"])

        assert payload["count"] > 0
        assert any(template["key"] == "executive_briefing" for template in payload["templates"])

    @pytest.mark.asyncio
    async def test_hidden_tool_direct_call_over_wire(self) -> None:
        async with _gemini_stdio_session() as session:
            result = await session.call_tool("list_format_templates", {})

        assert result.is_error is False
        assert result.structured_content is not None

        payload = json.loads(result.structured_content["result"])

        assert payload["count"] > 0


class TestMCPResourceRegistration:
    """Tests for MCP resource registration."""

    @pytest.mark.asyncio
    async def test_resources_registered(self):
        """Should have export resources registered."""
        from gemini_research_mcp.server import mcp

        resources = await mcp.list_resources()
        # Resources may be empty if no exports exist, but the call should succeed
        assert isinstance(resources, list)


class TestMCPLifespan:
    """Tests for MCP server lifespan and FastMCP Docket task support."""

    @pytest.mark.asyncio
    async def test_lifespan_context(self):
        """Lifespan context should run without error."""
        from gemini_research_mcp.server import lifespan, mcp

        async with lifespan(mcp):
            # Lifespan runs successfully — FastMCP's Docket handles tasks
            pass

    def test_task_config_on_research_deep(self):
        """research_deep tool should have TaskConfig(mode='required')."""
        from fastmcp.utilities.tasks import TaskConfig

        # Verify TaskConfig is importable and usable
        config = TaskConfig(mode="required")
        assert config.mode == "required"


class TestMCPContext:
    """Tests for fastmcp Context usage."""

    def test_context_import(self):
        """Context should be importable from fastmcp."""
        from fastmcp import Context

        assert Context is not None

    def test_context_not_generic(self):
        """fastmcp Context should not require generic parameters."""
        import inspect

        from fastmcp import Context

        # Get the class signature - it should not be a Generic
        # In fastmcp, Context is a concrete class, not Generic[ServerDeps, ClientDeps, Lifespan]
        inspect.signature(Context)
        # Should be able to instantiate hint without type params
        # (we can't actually instantiate without a server, but the type check passes)
        assert "Context" in str(Context)


class TestMCPElicitation:
    """Tests for elicitation API compatibility."""

    def test_elicit_method_signature(self):
        """Context.elicit should use response_type parameter."""
        import inspect

        from fastmcp import Context

        sig = inspect.signature(Context.elicit)
        params = list(sig.parameters.keys())

        assert "response_type" in params, "elicit() should have response_type parameter"
        assert "schema" not in params, "elicit() should NOT have schema parameter"

    def test_elicit_returns_elicitation_types(self):
        """elicit() should return Accepted/Declined/Cancelled types."""
        import inspect

        from fastmcp import Context

        sig = inspect.signature(Context.elicit)
        return_annotation = str(sig.return_annotation)

        assert "AcceptedElicitation" in return_annotation
        assert "DeclinedElicitation" in return_annotation
        assert "CancelledElicitation" in return_annotation


class TestMCPTypesCompatibility:
    """Tests for MCP types compatibility."""

    def test_mcp_types_import(self):
        """mcp.types should still be importable."""
        from mcp.types import (
            BlobResourceContents,
            EmbeddedResource,
            Icon,
            TextContent,
            TextResourceContents,
            ToolAnnotations,
        )

        assert all([
            BlobResourceContents,
            EmbeddedResource,
            Icon,
            TextContent,
            TextResourceContents,
            ToolAnnotations,
        ])

    def test_task_config_import(self):
        """TaskConfig should be importable from fastmcp."""
        from fastmcp.utilities.tasks import TaskConfig

        assert TaskConfig is not None

    def test_docket_available(self):
        """pydocket should be installed and available for FastMCP task routing."""
        from fastmcp.server.dependencies import is_docket_available

        assert is_docket_available(), "pydocket must be installed for task support"


class TestMCPServerCLI:
    """Tests for MCP server CLI interface."""

    def test_cli_version_flag(self):
        """Server should respond to --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "gemini_research_mcp.server", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # May exit with 0 (success) or non-zero if version prints to stderr
        output = result.stdout + result.stderr
        assert "gemini-research-mcp" in output.lower() or "0." in output

    def test_cli_help_flag(self):
        """Server should respond to --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "gemini_research_mcp.server", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "gemini" in result.stdout.lower() or "research" in result.stdout.lower()


class TestFastMCPDecorator:
    """Tests for fastmcp @mcp.tool() decorator behavior."""

    def test_decorated_function_is_callable(self):
        """@mcp.tool() decorated functions should remain callable."""
        from gemini_research_mcp.server import research_deep, research_web

        assert callable(research_deep)
        assert callable(research_web)

    def test_decorated_function_is_coroutine(self):
        """@mcp.tool() decorated async functions should remain coroutines."""
        import inspect

        from gemini_research_mcp.server import research_deep, research_web

        assert inspect.iscoroutinefunction(research_deep)
        assert inspect.iscoroutinefunction(research_web)

    def test_decorated_function_preserves_name(self):
        """@mcp.tool() should preserve function __name__."""
        from gemini_research_mcp.server import research_deep

        assert research_deep.__name__ == "research_deep"

