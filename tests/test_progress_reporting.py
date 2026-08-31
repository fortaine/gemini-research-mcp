import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_research_deep_emits_progress_for_thought(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.types import (
        DeepResearchAgent,
        DeepResearchProgress,
        DeepResearchResult,
    )

    captured: dict[str, DeepResearchAgent | None] = {}

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
        mcp_servers: list[dict[str, object]] | None = None,
        agent_name: DeepResearchAgent | None = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
    ) -> AsyncIterator[DeepResearchProgress]:
        del mcp_servers
        captured["agent_name"] = agent_name
        yield DeepResearchProgress(event_type="start", interaction_id="test-interaction")
        yield DeepResearchProgress(
            event_type="thought",
            interaction_id="test-interaction",
            content="Thinking about the answer",
        )

    async def fake_status(interaction_id: str) -> DeepResearchResult:
        return DeepResearchResult(
            text="Hello world",
            citations=[],
            thinking_summaries=[],
            interaction_id=interaction_id,
            usage=None,
            raw_interaction=types.SimpleNamespace(status="completed"),
        )

    async def passthrough_citations(
        result: DeepResearchResult,
        resolve_urls: bool,
    ) -> DeepResearchResult:
        return result

    async def fake_generate_title_from_query(query: str) -> str | None:
        return None

    async def fake_generate_session_metadata(text: str, query: str) -> Any:
        return types.SimpleNamespace(title=None, summary=None)

    def fake_save_research_session(**kwargs: Any) -> None:
        return None

    def fake_update_research_session(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "get_research_status", fake_status)
    monkeypatch.setattr(server, "process_citations", passthrough_citations)
    monkeypatch.setattr(server, "generate_title_from_query", fake_generate_title_from_query)
    monkeypatch.setattr(server, "generate_session_metadata", fake_generate_session_metadata)
    monkeypatch.setattr(server, "save_research_session", fake_save_research_session)
    monkeypatch.setattr(server, "update_research_session", fake_update_research_session)

    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.elicit = AsyncMock()

    result = await server.research_deep(query="test", ctx=ctx)

    assert "## Research Report" in result
    assert captured["agent_name"] == DeepResearchAgent.DEEP_RESEARCH

    # Start event is emitted through task statusMessage channel.
    ctx.report_progress.assert_any_await(
        progress=0,
        total=100,
        message="🚀 Research started",
    )

    ctx.report_progress.assert_any_await(
        progress=5,
        total=100,
        message="[1] 🧠 Thinking about the answer",
    )

    # Progress updates are unified on report_progress for task-mode execution.
    ctx.info.assert_not_awaited()


@pytest.mark.asyncio
async def test_research_deep_max_routes_max_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.types import (
        DeepResearchAgent,
        DeepResearchProgress,
        DeepResearchResult,
    )

    captured: dict[str, DeepResearchAgent | None] = {}

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
        mcp_servers: list[dict[str, object]] | None = None,
        agent_name: DeepResearchAgent | None = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
    ) -> AsyncIterator[DeepResearchProgress]:
        del mcp_servers
        captured["agent_name"] = agent_name
        yield DeepResearchProgress(event_type="start", interaction_id="test-interaction-max")

    async def fake_status(interaction_id: str) -> DeepResearchResult:
        return DeepResearchResult(
            text="Max report",
            citations=[],
            thinking_summaries=[],
            interaction_id=interaction_id,
            usage=None,
            raw_interaction=types.SimpleNamespace(status="completed"),
        )

    async def passthrough_citations(
        result: DeepResearchResult,
        resolve_urls: bool,
    ) -> DeepResearchResult:
        return result

    async def fake_generate_title_from_query(query: str) -> str | None:
        return "Max title"

    async def fake_generate_session_metadata(text: str, query: str) -> Any:
        return types.SimpleNamespace(title="Max title", summary="Max summary")

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "get_research_status", fake_status)
    monkeypatch.setattr(server, "process_citations", passthrough_citations)
    monkeypatch.setattr(server, "generate_title_from_query", fake_generate_title_from_query)
    monkeypatch.setattr(server, "generate_session_metadata", fake_generate_session_metadata)
    monkeypatch.setattr(server, "save_research_session", lambda **kwargs: None)
    monkeypatch.setattr(server, "update_research_session", lambda *args, **kwargs: None)

    result = await server.research_deep_max(query="high-stakes due diligence", ctx=None)

    assert "## Research Report" in result
    assert captured["agent_name"] == DeepResearchAgent.DEEP_RESEARCH_MAX


@pytest.mark.asyncio
async def test_research_deep_retryable_stream_error_stays_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import ResearchStatus
    from gemini_research_mcp.types import DeepResearchProgress

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
        mcp_servers: list[dict[str, object]] | None = None,
        agent_name: object | None = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
    ) -> AsyncIterator[DeepResearchProgress]:
        del query, format_instructions, file_search_store_names, mcp_servers, agent_name
        del visualization, collaborative_planning
        yield DeepResearchProgress(event_type="start", interaction_id="retryable-id")
        yield DeepResearchProgress(
            event_type="error",
            interaction_id="retryable-id",
            content="gateway_timeout: upstream gateway timed out",
        )

    updates: list[dict[str, Any]] = []

    async def fake_generate_title_from_query(query: str) -> str | None:
        return "Retryable timeout"

    def fake_save_research_session(**kwargs: Any) -> None:
        return None

    def fake_update_research_session(*args: Any, **kwargs: Any) -> None:
        del args
        updates.append(kwargs)

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "generate_title_from_query", fake_generate_title_from_query)
    monkeypatch.setattr(server, "save_research_session", fake_save_research_session)
    monkeypatch.setattr(server, "update_research_session", fake_update_research_session)

    with pytest.raises(server.DeepResearchError) as exc_info:
        await server.research_deep(query="test retryable timeout")

    assert exc_info.value.code == "RESEARCH_INTERRUPTED"
    assert "resume_research" in exc_info.value.message
    assert exc_info.value.details["interaction_id"] == "retryable-id"
    assert updates[-1]["status"] == ResearchStatus.INTERRUPTED


@pytest.mark.asyncio
async def test_research_deep_poll_timeout_returns_resume_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import ResearchStatus
    from gemini_research_mcp.types import DeepResearchProgress, DeepResearchResult

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
        mcp_servers: list[dict[str, object]] | None = None,
        agent_name: object | None = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
    ) -> AsyncIterator[DeepResearchProgress]:
        del query, format_instructions, file_search_store_names, mcp_servers, agent_name
        del visualization, collaborative_planning
        yield DeepResearchProgress(event_type="start", interaction_id="poll-timeout-id")

    async def fake_status(interaction_id: str) -> DeepResearchResult:
        return DeepResearchResult(
            text="",
            interaction_id=interaction_id,
            raw_interaction=types.SimpleNamespace(status="in_progress"),
        )

    updates: list[dict[str, Any]] = []

    async def fake_generate_title_from_query(query: str) -> str | None:
        return "Poll timeout"

    def fake_update_research_session(*args: Any, **kwargs: Any) -> None:
        del args
        updates.append(kwargs)

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "get_research_status", fake_status)
    monkeypatch.setattr(server, "generate_title_from_query", fake_generate_title_from_query)
    monkeypatch.setattr(server, "save_research_session", lambda **kwargs: None)
    monkeypatch.setattr(server, "update_research_session", fake_update_research_session)
    monkeypatch.setattr(server, "DEEP_RESEARCH_POLL_MAX_WAIT_SECONDS", 0)

    result = await server.research_deep(query="test poll timeout")

    assert "## Research Still Running" in result
    assert "poll-timeout-id" in result
    assert "resume_research" in result
    assert updates[-1]["status"] == ResearchStatus.INTERRUPTED


@pytest.mark.asyncio
async def test_research_deep_preserves_streamed_text_when_status_output_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import ResearchStatus
    from gemini_research_mcp.types import DeepResearchProgress, DeepResearchResult

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
        mcp_servers: list[dict[str, object]] | None = None,
        agent_name: object | None = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
    ) -> AsyncIterator[DeepResearchProgress]:
        del query, format_instructions, file_search_store_names, mcp_servers, agent_name
        del visualization, collaborative_planning
        yield DeepResearchProgress(event_type="start", interaction_id="stream-text-id")
        yield DeepResearchProgress(
            event_type="text",
            interaction_id="stream-text-id",
            content="Streamed report body.",
        )

    async def fake_status(interaction_id: str) -> DeepResearchResult:
        return DeepResearchResult(
            text="",
            interaction_id=interaction_id,
            raw_interaction=types.SimpleNamespace(status="completed"),
        )

    updates: list[dict[str, Any]] = []

    async def passthrough_citations(
        result: DeepResearchResult,
        resolve_urls: bool,
    ) -> DeepResearchResult:
        del resolve_urls
        return result

    async def fake_generate_metadata(text: str, query: str) -> Any:
        assert text == "Streamed report body."
        return types.SimpleNamespace(title="Streamed title", summary="Streamed summary")

    async def fake_generate_title(query: str) -> str:
        return "Initial title"

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "get_research_status", fake_status)
    monkeypatch.setattr(server, "process_citations", passthrough_citations)
    monkeypatch.setattr(server, "generate_title_from_query", fake_generate_title)
    monkeypatch.setattr(server, "generate_session_metadata", fake_generate_metadata)
    monkeypatch.setattr(server, "save_research_session", lambda **kwargs: None)
    monkeypatch.setattr(
        server,
        "update_research_session",
        lambda *args, **kwargs: updates.append(kwargs),
    )

    result = await server.research_deep(query="test streamed text")

    assert "Streamed report body." in result
    assert updates[-1]["report_text"] == "Streamed report body."
    assert updates[-1]["status"] == ResearchStatus.COMPLETED
