from types import SimpleNamespace
from typing import Any

import pytest

from gemini_research_mcp import deep
from gemini_research_mcp.deep import (
    _extract_text_from_interaction,
    _extract_usage,
    _validate_mcp_server_tool,
    analyze_mcp_tool_for_gemini,
    build_interactions_tools,
    deep_research_stream,
    validate_mcp_servers_supported,
)
from gemini_research_mcp.types import DeepResearchAgent, DeepResearchError, ErrorCategory


def _legacy_build_interactions_tools_combines_file_search_and_mcp() -> None:
    tools = build_interactions_tools(
        file_search_store_names=["fileSearchStores/market"],
        mcp_servers=[
            {
                "name": "Market Researcher MCP",
                "url": "https://mcp.example.com/mcp",
                "headers": {"Authorization": "Bearer secret"},
                "allowed_tools": ["market_get_mission", "market_generate_report"],
            }
        ],
    )

    assert tools == [
        {
            "type": "file_search",
            "file_search_store_names": ["fileSearchStores/market"],
        },
        {
            "type": "mcp_server",
            "name": "Market Researcher MCP",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer secret"},
            "allowed_tools": [{"tools": ["market_get_mission", "market_generate_report"]}],
        },
    ]


def test_build_interactions_tools_supports_file_search_without_mcp() -> None:
    assert build_interactions_tools(
        file_search_store_names=["fileSearchStores/market"],
    ) == [
        {
            "type": "file_search",
            "file_search_store_names": ["fileSearchStores/market"],
        }
    ]


def test_build_interactions_tools_rejects_remote_mcp() -> None:
    with pytest.raises(ValueError, match="Remote MCP is disabled"):
        build_interactions_tools(
            mcp_servers=[{"url": "https://mcp.example.com/mcp"}]
        )


def test_extract_text_from_interaction_steps_model_output() -> None:
    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="user_input",
                content=[SimpleNamespace(type="text", text="User prompt")],
            ),
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text="First report section.")],
            ),
            SimpleNamespace(
                type="model_output",
                content=[
                    SimpleNamespace(type="image"),
                    SimpleNamespace(type="text", text="Second report section."),
                ],
            ),
        ]
    )

    assert (
        _extract_text_from_interaction(interaction)
        == "First report section.\n\nSecond report section."
    )


def test_extract_text_from_interaction_prefers_output_text() -> None:
    interaction = SimpleNamespace(
        output_text="Official SDK output text.",
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text="Manual fallback text.")],
            ),
        ],
    )

    assert _extract_text_from_interaction(interaction) == "Official SDK output text."


def test_extract_text_from_interaction_dict_steps() -> None:
    interaction = {
        "steps": [
            {"type": "thought", "content": [{"type": "text", "text": "private thought"}]},
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Final dict report."}],
            },
        ]
    }

    assert _extract_text_from_interaction(interaction) == "Final dict report."


def test_extract_text_from_interaction_skips_reasoning_content() -> None:
    interaction = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "thinking", "text": "hidden reasoning"},
                    {"type": "text", "thought": True, "text": "hidden thought"},
                    {"type": "text", "text": "Visible final answer."},
                ],
            },
        ]
    }

    assert _extract_text_from_interaction(interaction) == "Visible final answer."


def test_extract_text_from_interaction_ignores_legacy_outputs_only_payload() -> None:
    interaction = SimpleNamespace(
        outputs=[
            SimpleNamespace(text="Legacy output one."),
            SimpleNamespace(content="Legacy output two."),
        ]
    )

    assert _extract_text_from_interaction(interaction) is None


def test_extract_usage_supports_interactions_usage_fields() -> None:
    interaction = SimpleNamespace(
        usage=SimpleNamespace(
            total_input_tokens=11,
            total_output_tokens=22,
            total_tokens=33,
        )
    )

    usage = _extract_usage(interaction)

    assert usage is not None
    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 22
    assert usage.total_tokens == 33


def test_validate_mcp_server_tool_rejects_non_https_remote_mcp() -> None:
    with pytest.raises(ValueError, match="must be HTTPS"):
        _validate_mcp_server_tool({"url": "http://mcp.example.com/mcp"})


def test_validate_mcp_server_tool_rejects_missing_mcp_url() -> None:
    with pytest.raises(ValueError, match="non-empty 'url'"):
        _validate_mcp_server_tool({"name": "Missing URL"})


def test_validate_mcp_server_tool_allows_explicit_localhost_dev_override() -> None:
    tool = _validate_mcp_server_tool(
        {"url": "http://127.0.0.1:8000/mcp", "allow_insecure_localhost": True}
    )

    assert tool == {"type": "mcp_server", "url": "http://127.0.0.1:8000/mcp"}


def test_validate_mcp_server_tool_validates_mcp_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _validate_mcp_server_tool(
            {"name": "", "url": "https://mcp.example.com/mcp"}
        )


def test_validate_mcp_server_tool_validates_headers() -> None:
    with pytest.raises(ValueError, match="headers"):
        _validate_mcp_server_tool({
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": 123},
        })


def test_validate_mcp_server_tool_validates_allowed_tools() -> None:
    with pytest.raises(ValueError, match="allowed_tools"):
        _validate_mcp_server_tool({
            "url": "https://mcp.example.com/mcp",
            "allowed_tools": ["ok", 42],
        })


def test_validate_mcp_servers_supported_rejects_standard_deep_research() -> None:
    with pytest.raises(ValueError, match="Remote MCP is disabled"):
        validate_mcp_servers_supported(
            agent_name=DeepResearchAgent.DEEP_RESEARCH,
            mcp_servers=[{"url": "https://mcp.example.com/mcp"}],
        )


def test_validate_mcp_servers_supported_rejects_deep_research_max() -> None:
    with pytest.raises(ValueError, match="Remote MCP is disabled"):
        validate_mcp_servers_supported(
            agent_name=DeepResearchAgent.DEEP_RESEARCH_MAX,
            mcp_servers=[{"url": "https://mcp.example.com/mcp"}],
        )


def test_validate_mcp_servers_supported_accepts_standard_without_mcp() -> None:
    validate_mcp_servers_supported(
        agent_name=DeepResearchAgent.DEEP_RESEARCH,
        mcp_servers=None,
    )


@pytest.mark.asyncio
async def test_deep_research_stream_rejects_mcp_on_standard_agent() -> None:
    stream = deep_research_stream(
        query="Use my MCP tool",
        agent_name=DeepResearchAgent.DEEP_RESEARCH,
        mcp_servers=[{"url": "https://mcp.example.com/mcp"}],
    )

    with pytest.raises(ValueError, match="Remote MCP is disabled"):
        await anext(stream)


def test_analyze_mcp_tool_for_gemini_accepts_simple_described_schema() -> None:
    issues = analyze_mcp_tool_for_gemini({
        "name": "get_guardrail_summary",
        "description": "Return a deterministic guardrail summary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to summarize.",
                }
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    })

    assert issues == []


def test_analyze_mcp_tool_for_gemini_flags_fixture_incompatible_schema() -> None:
    issues = analyze_mcp_tool_for_gemini({
        "name": "get_guardrail_summary",
        "description": None,
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    })

    assert "missing tool description" in issues
    assert "input schema has no properties; add at least one explicit argument" in issues


def test_analyze_mcp_tool_for_gemini_flags_complex_json_schema_keywords() -> None:
    issues = analyze_mcp_tool_for_gemini({
        "name": "complex_tool",
        "description": "Uses complex schema features.",
        "inputSchema": {
            "type": "object",
            "$defs": {"Filter": {"type": "object"}},
            "properties": {"filter": {"$ref": "#/$defs/Filter"}},
        },
    })

    assert any("$.$defs" in issue and "$.properties.filter.$ref" in issue for issue in issues)


@pytest.mark.asyncio
async def _legacy_deep_research_stream_passes_mcp_servers_to_interactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeStream:
        def __aiter__(self) -> "FakeStream":
            self._events = iter([
                SimpleNamespace(
                    event_type="interaction.start",
                    interaction=SimpleNamespace(id="interaction-fixture"),
                    event_id="event-start",
                ),
                SimpleNamespace(
                    event_type="interaction.complete",
                    interaction=SimpleNamespace(status="completed"),
                    event_id="event-complete",
                ),
            ])
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self._events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeInteractions:
        async def create(self, **kwargs: Any) -> FakeStream:
            captured.update(kwargs)
            return FakeStream()

    fake_client = SimpleNamespace(aio=SimpleNamespace(interactions=FakeInteractions()))
    monkeypatch.setattr(deep, "_get_healthy_client", lambda: fake_client)

    events = [
        event
        async for event in deep_research_stream(
            "Use the fixture MCP server.",
            agent_name=DeepResearchAgent.DEEP_RESEARCH_MAX,
            mcp_servers=[
                {
                    "name": "Fixture MCP",
                    "url": "https://fixture.example.com/mcp",
                    "allowed_tools": ["get_fixture"],
                }
            ],
        )
    ]

    assert [event.event_type for event in events] == ["start", "complete"]
    assert captured["agent"] == DeepResearchAgent.DEEP_RESEARCH_MAX.value
    assert captured["tools"] == [
        {
            "type": "mcp_server",
            "name": "Fixture MCP",
            "url": "https://fixture.example.com/mcp",
            "allowed_tools": [{"tools": ["get_fixture"]}],
        }
    ]


@pytest.mark.asyncio
async def test_deep_research_stream_rejects_max_mcp_before_client_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client_access() -> Any:
        raise AssertionError("Gemini client must not be accessed")

    monkeypatch.setattr(deep, "_get_healthy_client", fail_client_access)
    stream = deep_research_stream(
        "Use the fixture MCP server.",
        agent_name=DeepResearchAgent.DEEP_RESEARCH_MAX,
        mcp_servers=[{"url": "https://fixture.example.com/mcp"}],
    )

    with pytest.raises(ValueError, match="Remote MCP is disabled"):
        await anext(stream)


@pytest.mark.asyncio
async def test_server_rejects_mcp_before_clarification_or_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_research_mcp import server

    async def fail_clarification(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("Clarification must not run")

    def fail_stream(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Deep Research stream must not run")

    monkeypatch.setattr(server, "_maybe_clarify_query", fail_clarification)
    monkeypatch.setattr(server, "deep_research_stream", fail_stream)

    with pytest.raises(DeepResearchError) as exc_info:
        await server._run_deep_research_tool(
            query="Use private MCP data",
            mcp_servers=[
                {
                    "url": "https://user:secret@example.com/mcp?token=secret",
                    "headers": {"Authorization": "Bearer secret"},
                }
            ],
            agent_name=DeepResearchAgent.DEEP_RESEARCH_MAX,
            tool_name="research_deep_max",
        )

    error = exc_info.value
    assert error.code == "REMOTE_MCP_DISABLED"
    assert error.category == ErrorCategory.UNSUPPORTED_FEATURE
    assert "secret" not in str(error)
    assert error.details["diagnostic_tool"] == "inspect_mcp_server_for_gemini"


@pytest.mark.asyncio
async def test_deep_research_stream_maps_genai_v2_step_delta_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream():
        yield SimpleNamespace(
            event_type="interaction.created",
            interaction=SimpleNamespace(id="interaction-v2"),
            event_id="event-start",
        )
        yield SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(
                type="thought_summary",
                content=SimpleNamespace(text="Plan search strategy"),
            ),
            event_id="event-thought",
        )
        yield SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="text", text="Final report paragraph."),
            event_id="event-text",
        )
        yield SimpleNamespace(
            event_type="interaction.completed",
            interaction=SimpleNamespace(id="interaction-v2", status="completed"),
            event_id="event-complete",
        )

    class FakeInteractions:
        async def create(self, **kwargs: Any):
            del kwargs
            return fake_stream()

    fake_client = SimpleNamespace(aio=SimpleNamespace(interactions=FakeInteractions()))
    monkeypatch.setattr(deep, "_get_healthy_client", lambda: fake_client)

    events = [
        event
        async for event in deep_research_stream(
            "Use the GenAI 2.x stream shape.",
        )
    ]

    assert [event.event_type for event in events] == ["start", "thought", "text", "complete"]
    assert events[0].interaction_id == "interaction-v2"
    assert events[1].content == "Plan search strategy"
    assert events[2].content == "Final report paragraph."


@pytest.mark.asyncio
async def test_deep_research_stream_maps_genai_v2_status_failure_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream():
        yield SimpleNamespace(
            event_type="interaction.created",
            interaction=SimpleNamespace(id="interaction-failed"),
            event_id="event-start",
        )
        yield SimpleNamespace(
            event_type="interaction.status_update",
            interaction_id="interaction-failed",
            status="failed",
            event_id="event-failed",
        )

    class FakeInteractions:
        async def create(self, **kwargs: Any):
            del kwargs
            return fake_stream()

    fake_client = SimpleNamespace(aio=SimpleNamespace(interactions=FakeInteractions()))
    monkeypatch.setattr(deep, "_get_healthy_client", lambda: fake_client)

    events = [
        event
        async for event in deep_research_stream(
            "Map provider status failure.",
        )
    ]

    assert [event.event_type for event in events] == ["start", "error"]
    assert events[1].interaction_id == "interaction-failed"
    assert "failed" in (events[1].content or "")
