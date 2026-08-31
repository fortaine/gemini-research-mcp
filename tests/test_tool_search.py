"""Tests for the FastMCP BM25 tool-search surface."""

import json

import pytest


class TestToolSearchTransform:
    """Validate the visible tool surface and synthetic search tools."""

    @pytest.mark.asyncio
    async def test_visible_tools_are_pinned_and_synthetic(self) -> None:
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_names = {tool.name for tool in tools}

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
        from gemini_research_mcp.server import mcp

        result = await mcp.call_tool("search_tools", {"query": "fetch webpage content"})

        assert result.structured_content is not None
        matches = result.structured_content["result"]
        fetch_tool = next(tool for tool in matches if tool["name"] == "fetch_webpage")

        assert "url" in fetch_tool["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_search_tools_excludes_pinned_tools(self) -> None:
        from gemini_research_mcp.server import mcp

        result = await mcp.call_tool("search_tools", {"query": "research"})

        assert result.structured_content is not None
        matches = {tool["name"] for tool in result.structured_content["result"]}

        assert "research_web" not in matches
        assert "research_deep" not in matches
        assert "resume_research" not in matches
        assert "export_research_session" not in matches

    @pytest.mark.asyncio
    async def test_call_tool_proxy_invokes_hidden_tool(self) -> None:
        from gemini_research_mcp.server import mcp

        result = await mcp.call_tool(
            "call_tool",
            {"name": "list_format_templates", "arguments": {}},
        )

        assert result.structured_content is not None
        payload = json.loads(result.structured_content["result"])

        assert payload["count"] > 0
        assert any(template["key"] == "executive_briefing" for template in payload["templates"])

    @pytest.mark.asyncio
    async def test_hidden_tools_remain_directly_callable(self) -> None:
        from gemini_research_mcp.server import mcp

        result = await mcp.call_tool("list_format_templates", {})

        assert result.structured_content is not None
        payload = json.loads(result.structured_content["result"])

        assert payload["count"] > 0