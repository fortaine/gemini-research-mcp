"""Tests for chantier 7: visualization, image deltas, and collaborative planning.

Covers:
- `visualization`/`collaborative_planning` threading into the Interactions API
  `agent_config` payload (deep.py -> deep_research_stream).
- Image content/delta items are distinguished from text (never concatenated
  into the report) both in the streaming parser and in
  `_extract_text_from_interaction`/`_extract_images_from_interaction`.
- Image artifact persistence + `research://exports/{id}` resource link exposure
  (server.py's `_persist_deep_research_image(s)`), using an isolated export
  store so tests never touch the real XDG data dir.
- The collaborative-planning guard-pattern flow: research_deep(...,
  collaborative_planning=True) -> plan + interaction_id, then
  refine_research_plan(...) to iterate or approve.
- The extended `ResearchStatus` enum (no regressions to existing members).
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point session + export storage singletons at a tmp_path-backed store.

    Mirrors the isolated-store fixture pattern used in tests/test_export.py
    and tests/test_cancellation.py so these tests never write to the real
    XDG data dir.
    """
    import gemini_research_mcp.storage as storage_module

    session_store = storage_module.SessionStorage(storage_dir=tmp_path / "sessions")
    export_store = storage_module.ExportArtifactStore(storage_dir=tmp_path / "exports")

    monkeypatch.setattr(storage_module, "_storage", session_store)
    monkeypatch.setattr(storage_module, "_export_store", export_store)

    import gemini_research_mcp.server as server_module

    monkeypatch.setattr(server_module, "get_export_store", lambda: export_store)

    yield session_store, export_store


def _chunk(**kwargs: Any) -> SimpleNamespace:
    """Build a fake google-genai interactions stream chunk."""
    return SimpleNamespace(event_id=None, **kwargs)


class _FakeInteractionsClient:
    """Fake `client.aio.interactions` that replays a fixed list of chunks."""

    def __init__(self, chunks: list[Any], captured_kwargs: dict[str, Any]) -> None:
        self._chunks = chunks
        self._captured_kwargs = captured_kwargs

    async def create(self, **kwargs: Any) -> AsyncIterator[Any]:
        self._captured_kwargs.update(kwargs)

        async def _gen() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _install_fake_stream(
    monkeypatch: pytest.MonkeyPatch, chunks: list[Any]
) -> dict[str, Any]:
    """Monkeypatch deep._get_healthy_client to replay `chunks`; return captured kwargs."""
    import gemini_research_mcp.deep as deep_module

    captured_kwargs: dict[str, Any] = {}
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(interactions=_FakeInteractionsClient(chunks, captured_kwargs))
    )
    monkeypatch.setattr(deep_module, "_get_healthy_client", lambda: fake_client)
    monkeypatch.setattr(deep_module, "_record_client_success", lambda: None)
    monkeypatch.setattr(deep_module, "_record_client_failure", lambda: None)
    return captured_kwargs


# =============================================================================
# 1. `visualization` / `collaborative_planning` threading into agent_config
# =============================================================================


@pytest.mark.asyncio
async def test_visualization_and_collaborative_planning_thread_into_agent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_research_mcp.deep import deep_research_stream
    from gemini_research_mcp.types import DeepResearchAgent

    chunks = [
        _chunk(
            event_type="interaction.created",
            interaction="test-visualization-1",
            interaction_id="test-visualization-1",
        ),
        _chunk(event_type="interaction.completed", interaction={"status": "completed"}),
    ]
    captured = _install_fake_stream(monkeypatch, chunks)

    events = [
        event
        async for event in deep_research_stream(
            "test query",
            agent_name=DeepResearchAgent.DEEP_RESEARCH,
            visualization="auto",
            collaborative_planning=True,
        )
    ]

    assert any(e.event_type == "start" for e in events)
    agent_config = captured["agent_config"]
    assert agent_config["visualization"] == "auto"
    assert agent_config["collaborative_planning"] is True


@pytest.mark.asyncio
async def test_visualization_defaults_to_off_and_planning_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_research_mcp.deep import deep_research_stream
    from gemini_research_mcp.types import DeepResearchAgent

    chunks = [
        _chunk(
            event_type="interaction.created",
            interaction="test-visualization-2",
            interaction_id="test-visualization-2",
        ),
        _chunk(event_type="interaction.completed", interaction={"status": "completed"}),
    ]
    captured = _install_fake_stream(monkeypatch, chunks)

    _ = [
        event
        async for event in deep_research_stream(
            "test query", agent_name=DeepResearchAgent.DEEP_RESEARCH
        )
    ]

    agent_config = captured["agent_config"]
    assert agent_config["visualization"] == "off"
    assert agent_config["collaborative_planning"] is False


@pytest.mark.asyncio
async def test_previous_interaction_id_threads_into_create_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_research_mcp.deep import deep_research_stream
    from gemini_research_mcp.types import DeepResearchAgent

    chunks = [
        _chunk(
            event_type="interaction.created",
            interaction="test-continue-1",
            interaction_id="test-continue-1",
        ),
        _chunk(event_type="interaction.completed", interaction={"status": "completed"}),
    ]
    captured = _install_fake_stream(monkeypatch, chunks)

    _ = [
        event
        async for event in deep_research_stream(
            "approve the plan",
            agent_name=DeepResearchAgent.DEEP_RESEARCH,
            previous_interaction_id="parent-interaction-id",
        )
    ]

    assert captured["previous_interaction_id"] == "parent-interaction-id"


# =============================================================================
# 2. Image deltas are never merged into text
# =============================================================================


@pytest.mark.asyncio
async def test_image_delta_yields_dedicated_image_event_not_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_research_mcp.deep import deep_research_stream
    from gemini_research_mcp.types import DeepResearchAgent

    chunks = [
        _chunk(
            event_type="interaction.created",
            interaction="test-image-1",
            interaction_id="test-image-1",
        ),
        _chunk(
            event_type="step.delta",
            delta={"type": "text", "text": "Here is the analysis: "},
        ),
        _chunk(
            event_type="step.delta",
            delta={
                "type": "image",
                "data": base64.b64encode(b"fake-png-bytes").decode("ascii"),
                "mime_type": "image/png",
                "uri": None,
            },
        ),
        _chunk(
            event_type="step.delta",
            delta={"type": "text", "text": "and the conclusion."},
        ),
        _chunk(event_type="interaction.completed", interaction={"status": "completed"}),
    ]
    _install_fake_stream(monkeypatch, chunks)

    events = [
        event
        async for event in deep_research_stream(
            "test query", agent_name=DeepResearchAgent.DEEP_RESEARCH
        )
    ]

    text_events = [e for e in events if e.event_type == "text"]
    image_events = [e for e in events if e.event_type == "image"]

    assert [e.content for e in text_events] == [
        "Here is the analysis: ",
        "and the conclusion.",
    ]
    assert len(image_events) == 1
    image_event = image_events[0]
    assert image_event.image_mime_type == "image/png"
    assert image_event.image_uri is None
    assert image_event.content == base64.b64encode(b"fake-png-bytes").decode("ascii")
    # The image payload must never appear inside a "text" event's content.
    assert all("fake-png-bytes" not in (e.content or "") for e in text_events)


def test_extract_text_from_interaction_skips_image_content() -> None:
    from gemini_research_mcp.deep import _extract_text_from_interaction

    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[
                    {"type": "text", "text": "Report body."},
                    {
                        "type": "image",
                        "data": base64.b64encode(b"img").decode("ascii"),
                        "mime_type": "image/png",
                    },
                ],
            )
        ]
    )

    text = _extract_text_from_interaction(interaction)

    assert text == "Report body."
    assert "img" not in (text or "")


def test_extract_images_from_interaction_returns_normalized_dicts() -> None:
    from gemini_research_mcp.deep import _extract_images_from_interaction

    encoded = base64.b64encode(b"img-bytes").decode("ascii")
    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[
                    {"type": "text", "text": "ignored for images"},
                    {"type": "image", "data": encoded, "mime_type": "image/jpeg", "uri": None},
                ],
            )
        ]
    )

    images = _extract_images_from_interaction(interaction)

    assert images == [{"data": encoded, "mime_type": "image/jpeg", "uri": None}]


def test_extract_images_from_interaction_empty_when_no_images() -> None:
    from gemini_research_mcp.deep import _extract_images_from_interaction

    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[{"type": "text", "text": "just text"}],
            )
        ]
    )

    assert _extract_images_from_interaction(interaction) == []


# =============================================================================
# 3. Image artifact persistence + resource link exposure
# =============================================================================


@pytest.mark.asyncio
async def test_persist_deep_research_image_with_inline_data(isolated_storage) -> None:
    from gemini_research_mcp.server import _persist_deep_research_image

    _, export_store = isolated_storage
    encoded = base64.b64encode(b"png-bytes").decode("ascii")

    export_id, resource_uri = await _persist_deep_research_image(
        session_id="session-1",
        index=0,
        data_b64=encoded,
        mime_type="image/png",
        uri=None,
    )

    assert export_id is not None
    assert resource_uri == f"research://exports/{export_id}"

    artifact = await export_store.get_async(export_id)
    assert artifact is not None
    assert artifact.mime_type == "image/png"
    assert artifact.content == b"png-bytes"
    assert artifact.session_id == "session-1"


@pytest.mark.asyncio
async def test_persist_deep_research_image_with_only_uri_is_not_reified() -> None:
    from gemini_research_mcp.server import _persist_deep_research_image

    export_id, resource_uri = await _persist_deep_research_image(
        session_id="session-1",
        index=0,
        data_b64=None,
        mime_type="image/png",
        uri="https://example.com/chart.png",
    )

    assert export_id is None
    assert resource_uri == "https://example.com/chart.png"


@pytest.mark.asyncio
async def test_persist_deep_research_images_batch(isolated_storage) -> None:
    from gemini_research_mcp.server import _persist_deep_research_images

    encoded_a = base64.b64encode(b"a").decode("ascii")
    encoded_b = base64.b64encode(b"b").decode("ascii")

    export_ids, resource_uris = await _persist_deep_research_images(
        session_id="session-2",
        images=[
            {"data": encoded_a, "mime_type": "image/png", "uri": None},
            {"data": encoded_b, "mime_type": "image/jpeg", "uri": None},
        ],
    )

    assert len(export_ids) == 2
    assert len(resource_uris) == 2
    assert all(uri.startswith("research://exports/") for uri in resource_uris)


def test_format_deep_research_report_includes_images_section() -> None:
    from gemini_research_mcp.server import _format_deep_research_report
    from gemini_research_mcp.types import DeepResearchResult

    result = DeepResearchResult(text="Body text", citations=[], thinking_summaries=[])

    report = _format_deep_research_report(
        result, "interaction-x", 12.0, image_uris=["research://exports/abc123"]
    )

    assert "## Images" in report
    assert "research://exports/abc123" in report


def test_export_markdown_includes_image_links(isolated_storage) -> None:
    from gemini_research_mcp.export import _format_markdown_export
    from gemini_research_mcp.storage import ResearchSession

    session = ResearchSession(
        interaction_id="session-3",
        query="test query",
        created_at=0.0,
        report_text="the report",
        image_export_ids=["img-1", "img-2"],
    )

    markdown = _format_markdown_export(session)

    assert "## Images" in markdown
    assert "research://exports/img-1" in markdown
    assert "research://exports/img-2" in markdown


def test_export_json_includes_image_uris() -> None:
    from gemini_research_mcp.export import _session_to_export_dict
    from gemini_research_mcp.storage import ResearchSession

    session = ResearchSession(
        interaction_id="session-4",
        query="test query",
        created_at=0.0,
        report_text="the report",
        image_export_ids=["img-3"],
    )

    data = _session_to_export_dict(session)

    assert data["images"] == ["research://exports/img-3"]


# =============================================================================
# 4. Collaborative planning guard-pattern flow
# =============================================================================


def test_format_plan_response_includes_refine_hint() -> None:
    from gemini_research_mcp.server import _format_plan_response

    response = _format_plan_response("Step 1: search. Step 2: synthesize.", "plan-interaction-1")

    assert "Step 1: search." in response
    assert "plan-interaction-1" in response
    assert "refine_research_plan" in response
    assert 'decision="approve"' in response


@pytest.mark.asyncio
async def test_research_deep_collaborative_planning_returns_plan_awaiting_approval(
    monkeypatch: pytest.MonkeyPatch, isolated_storage
) -> None:
    """research_deep(collaborative_planning=True) returns the plan, not a report."""
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import ResearchStatus, get_research_session
    from gemini_research_mcp.types import DeepResearchProgress

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
        mcp_servers: list[dict[str, object]] | None = None,
        agent_name: Any = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
        previous_interaction_id: str | None = None,
    ) -> AsyncIterator[DeepResearchProgress]:
        del (
            format_instructions,
            file_search_store_names,
            mcp_servers,
            previous_interaction_id,
        )
        assert collaborative_planning is True
        assert visualization == "off"
        yield DeepResearchProgress(event_type="start", interaction_id="plan-interaction-2")
        yield DeepResearchProgress(
            event_type="text",
            interaction_id="plan-interaction-2",
            content=f"Plan for: {query}",
        )
        yield DeepResearchProgress(event_type="plan_ready", interaction_id="plan-interaction-2")

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(
        server, "generate_title_from_query", AsyncMockReturning("Plan title")
    )

    result = await server.research_deep(
        query="Investigate quantum error correction",
        collaborative_planning=True,
    )

    assert isinstance(result, str)
    assert "Research Plan (awaiting approval)" in result
    assert "plan-interaction-2" in result
    assert "Plan for: Investigate quantum error correction" in result

    session = get_research_session("plan-interaction-2")
    assert session is not None
    assert session.status == ResearchStatus.AWAITING_APPROVAL
    assert session.plan_text == "Plan for: Investigate quantum error correction"


@pytest.mark.asyncio
async def test_refine_research_plan_iterate_returns_new_plan(
    monkeypatch: pytest.MonkeyPatch, isolated_storage
) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import (
        ResearchStatus,
        get_research_session,
        save_research_session,
    )
    from gemini_research_mcp.types import DeepResearchAgent, DeepResearchProgress

    save_research_session(
        interaction_id="plan-interaction-3",
        query="Investigate quantum error correction",
        title="Plan title",
        agent_name=DeepResearchAgent.DEEP_RESEARCH,
        status=ResearchStatus.AWAITING_APPROVAL,
        plan_text="Initial plan",
    )

    async def fake_stream(
        *,
        query: str,
        agent_name: Any = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
        previous_interaction_id: str | None = None,
        **_: Any,
    ) -> AsyncIterator[DeepResearchProgress]:
        del agent_name, visualization
        assert collaborative_planning is True
        assert previous_interaction_id == "plan-interaction-3"
        yield DeepResearchProgress(event_type="start", interaction_id="plan-interaction-4")
        yield DeepResearchProgress(
            event_type="text", interaction_id="plan-interaction-4", content=f"Revised: {query}"
        )
        yield DeepResearchProgress(event_type="plan_ready", interaction_id="plan-interaction-4")

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)

    result = await server.refine_research_plan(
        previous_interaction_id="plan-interaction-3",
        decision="iterate",
        instructions="Please add a section on error thresholds.",
    )

    assert "Research Plan (awaiting approval)" in result
    assert "plan-interaction-4" in result
    assert "Revised: Please add a section on error thresholds." in result

    new_session = get_research_session("plan-interaction-4")
    assert new_session is not None
    assert new_session.status == ResearchStatus.AWAITING_APPROVAL
    assert new_session.plan_text == "Revised: Please add a section on error thresholds."


@pytest.mark.asyncio
async def test_refine_research_plan_approve_executes_and_returns_report(
    monkeypatch: pytest.MonkeyPatch, isolated_storage
) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import (
        ResearchStatus,
        get_research_session,
        save_research_session,
    )
    from gemini_research_mcp.types import (
        DeepResearchAgent,
        DeepResearchProgress,
        DeepResearchResult,
    )

    save_research_session(
        interaction_id="plan-interaction-5",
        query="Investigate quantum error correction",
        title="Plan title",
        agent_name=DeepResearchAgent.DEEP_RESEARCH,
        status=ResearchStatus.AWAITING_APPROVAL,
        plan_text="Approved plan",
    )

    async def fake_stream(
        *,
        query: str,
        agent_name: Any = None,
        visualization: str = "off",
        collaborative_planning: bool = False,
        previous_interaction_id: str | None = None,
        **_: Any,
    ) -> AsyncIterator[DeepResearchProgress]:
        del query, agent_name, visualization
        assert collaborative_planning is False
        assert previous_interaction_id == "plan-interaction-5"
        yield DeepResearchProgress(event_type="start", interaction_id="plan-interaction-6")
        yield DeepResearchProgress(
            event_type="text", interaction_id="plan-interaction-6", content="partial "
        )

    async def fake_status(interaction_id: str) -> DeepResearchResult:
        assert interaction_id == "plan-interaction-6"
        return DeepResearchResult(
            text="Final report body.",
            citations=[],
            thinking_summaries=[],
            interaction_id=interaction_id,
            raw_interaction=SimpleNamespace(status="completed"),
        )

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "get_research_status", fake_status)
    monkeypatch.setattr(
        server,
        "process_citations",
        AsyncMockReturningArg(),
    )
    monkeypatch.setattr(
        server,
        "generate_session_metadata",
        AsyncMockReturning(SimpleNamespace(title="", summary="")),
    )

    result = await server.refine_research_plan(
        previous_interaction_id="plan-interaction-5",
        decision="approve",
    )

    assert "## Research Report" in result
    assert "Final report body." in result

    final_session = get_research_session("plan-interaction-6")
    assert final_session is not None
    assert final_session.status == ResearchStatus.COMPLETED
    assert final_session.report_text == "Final report body."


@pytest.mark.asyncio
async def test_refine_research_plan_unknown_session_returns_error() -> None:
    import gemini_research_mcp.server as server

    result = await server.refine_research_plan(
        previous_interaction_id="does-not-exist",
        decision="approve",
    )

    assert "No research session found" in result


@pytest.mark.asyncio
async def test_refine_research_plan_wrong_status_returns_error(isolated_storage) -> None:
    import gemini_research_mcp.server as server
    from gemini_research_mcp.storage import ResearchStatus, save_research_session
    from gemini_research_mcp.types import DeepResearchAgent

    save_research_session(
        interaction_id="completed-session",
        query="q",
        title="t",
        agent_name=DeepResearchAgent.DEEP_RESEARCH,
        status=ResearchStatus.COMPLETED,
    )

    result = await server.refine_research_plan(
        previous_interaction_id="completed-session",
        decision="approve",
    )

    assert "not awaiting plan approval" in result


class AsyncMockReturning:
    """Simple async callable returning a fixed value, avoiding unittest.mock import churn."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._value


class AsyncMockReturningArg:
    """Async callable that returns its (first positional) argument unchanged."""

    async def __call__(self, arg: Any, *_args: Any, **_kwargs: Any) -> Any:
        return arg


# =============================================================================
# 5. Extended ResearchStatus enum - no regressions
# =============================================================================


def test_research_status_new_members_exist_and_are_distinct() -> None:
    from gemini_research_mcp.storage import ResearchStatus

    assert ResearchStatus.PLANNING.value == "planning"
    assert ResearchStatus.AWAITING_APPROVAL.value == "awaiting_approval"
    assert ResearchStatus.EXECUTING.value == "executing"

    values = [s.value for s in ResearchStatus]
    assert len(values) == len(set(values)), "ResearchStatus values must stay unique"


def test_research_status_existing_members_unchanged() -> None:
    from gemini_research_mcp.storage import ResearchStatus

    assert ResearchStatus.IN_PROGRESS.value == "in_progress"
    assert ResearchStatus.COMPLETED.value == "completed"
    assert ResearchStatus.FAILED.value == "failed"
    assert ResearchStatus.INTERRUPTED.value == "interrupted"
    assert ResearchStatus.CANCELLED.value == "cancelled"


def test_research_session_defaults_plan_text_and_image_export_ids() -> None:
    from gemini_research_mcp.storage import ResearchSession

    session = ResearchSession(interaction_id="x", query="q", created_at=0.0)

    assert session.plan_text is None
    assert session.image_export_ids == []
