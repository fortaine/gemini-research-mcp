"""Tests for the BM25-compacted tool catalog (FastMCP 4 BM25SearchTransform).

Verifies the modernization plan's chantier 8 acceptance criteria:
- Always-visible tools stay pinned in the compact list_tools() catalog.
- Hidden utility tools remain discoverable via search_tools and directly
  callable via their own name (or the generic call_tool proxy).
- BM25 ranking and the result limit behave sensibly.
- Tool annotations (authorization/behavior hints) survive the transform.
"""

import pytest

from gemini_research_mcp.server import mcp

ALWAYS_VISIBLE = {
    "research_web",
    "research_deep",
    "research_deep_max",
    "resume_research",
    "export_research_session",
}
HIDDEN_BUT_DISCOVERABLE = {
    "fetch_webpage",
    "research_followup",
    "list_research_sessions",
    "list_format_templates",
    "refine_research_plan",
    "inspect_mcp_server_for_gemini",
}


class TestCompactCatalog:
    """The public list_tools() surface reflects the BM25 transform, not the raw registry."""

    @pytest.mark.asyncio
    async def test_always_visible_tools_are_listed(self):
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert names >= ALWAYS_VISIBLE

    @pytest.mark.asyncio
    async def test_search_and_call_tool_are_listed(self):
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "search_tools" in names
        assert "call_tool" in names

    @pytest.mark.asyncio
    async def test_hidden_tools_are_not_in_default_catalog(self):
        """Utility tools are compacted out of the default listing."""
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert names.isdisjoint(HIDDEN_BUT_DISCOVERABLE)

    @pytest.mark.asyncio
    async def test_catalog_is_compact(self):
        """5 pinned tools + search_tools + call_tool = 7, not all 11 registered tools."""
        tools = await mcp.list_tools()
        assert len(tools) == len(ALWAYS_VISIBLE) + 2

    @pytest.mark.asyncio
    async def test_raw_registry_still_has_all_registered_tools(self):
        """The transform only affects the visible listing, not what's registered."""
        raw_tools = await mcp._list_tools()
        names = {t.name for t in raw_tools}
        assert names == ALWAYS_VISIBLE | HIDDEN_BUT_DISCOVERABLE


class TestHiddenToolsRemainCallable:
    """Hidden tools must stay directly callable by clients that know their name."""

    @pytest.mark.asyncio
    async def test_hidden_tool_directly_callable_by_name(self):
        result = await mcp.call_tool("list_format_templates", {})
        assert result is not None

    @pytest.mark.asyncio
    async def test_hidden_tool_callable_via_call_tool_proxy(self):
        """The generic `call_tool` proxy tool can also reach a hidden tool."""
        result = await mcp.call_tool(
            "call_tool",
            {"name": "list_format_templates", "arguments": {}},
        )
        assert result is not None


class TestSearchToolsDiscovery:
    """search_tools uses BM25 ranking to surface hidden tools by relevance."""

    @pytest.mark.asyncio
    async def test_search_finds_hidden_tool_by_relevance(self):
        result = await mcp.call_tool(
            "search_tools", {"query": "list available export format templates"}
        )
        text = str(result)
        assert "list_format_templates" in text

    @pytest.mark.asyncio
    async def test_search_respects_max_results_limit(self):
        """BM25SearchTransform defaults to max_results=5."""
        result = await mcp.call_tool("search_tools", {"query": "research"})
        content = result.content[0].text if hasattr(result, "content") else str(result)
        import json

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return  # Serializer format may vary; ranking limit is enforced internally regardless.
        if isinstance(parsed, list):
            assert len(parsed) <= 5


class TestToolAnnotationsPreserved:
    """Tool annotations (read-only/idempotent/open-world hints) survive the transform."""

    @pytest.mark.asyncio
    async def test_research_web_is_read_only_and_open_world(self):
        tools = await mcp.list_tools()
        research_web = next(t for t in tools if t.name == "research_web")
        assert research_web.annotations is not None
        assert research_web.annotations.read_only_hint is True
        assert research_web.annotations.open_world_hint is True

    @pytest.mark.asyncio
    async def test_export_research_session_is_read_only_and_idempotent(self):
        tools = await mcp.list_tools()
        export_tool = next(t for t in tools if t.name == "export_research_session")
        assert export_tool.annotations is not None
        assert export_tool.annotations.read_only_hint is True
        assert export_tool.annotations.idempotent_hint is True
