"""Tests for stateful quick research and its follow-up contract."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gemini_research_mcp.deep import research_followup
from gemini_research_mcp.quick import quick_research
from gemini_research_mcp.types import ResearchResult, Source


@pytest.mark.asyncio
async def test_quick_research_creates_interaction_and_extracts_grounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quick research should expose a stateful Interaction with Google Search evidence."""
    created: dict[str, object] = {}
    interaction = SimpleNamespace(
        id="quick-interaction-123",
        output_text="Windows Server 2016 remains supported in this example.",
        steps=[
            SimpleNamespace(
                type="google_search_call",
                arguments=SimpleNamespace(queries=["project X Windows Server 2016 support"]),
            ),
            SimpleNamespace(
                type="model_output",
                content=[
                    SimpleNamespace(
                        type="text",
                        text="Windows Server 2016 remains supported in this example.",
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                url="https://example.com/support",
                                title="Project X support policy",
                            )
                        ],
                    )
                ],
            ),
        ],
    )

    async def fake_create(**kwargs: object) -> object:
        created.update(kwargs)
        return interaction

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(interactions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr("gemini_research_mcp.quick.get_api_key", lambda: "test-key")
    monkeypatch.setattr("gemini_research_mcp.quick.get_model", lambda: "gemini-test")
    monkeypatch.setattr("gemini_research_mcp.quick.genai.Client", lambda api_key: fake_client)

    result = await quick_research("What support does Project X provide?")

    assert created["input"] == "What support does Project X provide?"
    assert created["model"] == "gemini-test"
    assert created["tools"] == [{"type": "google_search"}]
    assert result.interaction_id == "quick-interaction-123"
    assert result.text == "Windows Server 2016 remains supported in this example."
    assert result.queries == ["project X Windows Server 2016 support"]
    assert result.sources == [
        Source(uri="https://example.com/support", title="Project X support policy")
    ]


@pytest.mark.asyncio
async def test_quick_research_preserves_thinking_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactions should expose thought summaries when requested."""
    created: dict[str, object] = {}
    interaction = SimpleNamespace(
        id="quick-interaction-thoughts",
        output_text="Grounded answer.",
        steps=[
            SimpleNamespace(
                type="thought",
                summary=[SimpleNamespace(type="text", text="Compared current sources.")],
            )
        ],
    )

    async def fake_create(**kwargs: object) -> object:
        created.update(kwargs)
        return interaction

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(interactions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr("gemini_research_mcp.quick.get_api_key", lambda: "test-key")
    monkeypatch.setattr("gemini_research_mcp.quick.get_model", lambda: "gemini-test")
    monkeypatch.setattr("gemini_research_mcp.quick.genai.Client", lambda api_key: fake_client)

    result = await quick_research("Summarize the current sources.", include_thoughts=True)

    assert created["generation_config"] == {
        "thinking_level": "high",
        "thinking_summaries": "auto",
    }
    assert result.thinking_summary == "Compared current sources."


@pytest.mark.asyncio
async def test_followup_chains_interaction_and_reenables_google_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each follow-up must chain from the prior interaction and re-enable search."""
    created: list[dict[str, object]] = []
    responses = iter([
        SimpleNamespace(
            id="followup-interaction-456",
            output_text="A recent issue contradicts the earlier result.",
        ),
        SimpleNamespace(
            id="followup-interaction-789",
            output_text="The contradiction is limited to the legacy deployment.",
        ),
    ])

    async def fake_create(**kwargs: object) -> object:
        created.append(kwargs)
        return next(responses)

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(interactions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr("gemini_research_mcp.deep._get_healthy_client", lambda: fake_client)

    result = await research_followup(
        previous_interaction_id="quick-interaction-123",
        query="Search for issues that contradict this.",
        model="gemini-test",
        include_interaction_id=True,
    )
    assert isinstance(result, tuple)
    assert isinstance(result[1], str)
    next_result = await research_followup(
        previous_interaction_id=result[1],
        query="Explain the scope of that contradiction.",
        model="gemini-test",
        include_interaction_id=True,
    )

    assert result == (
        "A recent issue contradicts the earlier result.",
        "followup-interaction-456",
    )
    assert next_result == (
        "The contradiction is limited to the legacy deployment.",
        "followup-interaction-789",
    )
    assert [call["previous_interaction_id"] for call in created] == [
        "quick-interaction-123",
        "followup-interaction-456",
    ]
    assert all(call["tools"] == [{"type": "google_search"}] for call in created)


@pytest.mark.asyncio
async def test_followup_default_return_remains_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python helper keeps its historical string return by default."""
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            interactions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        id="followup-interaction-456",
                        output_text="Follow-up answer.",
                    )
                )
            )
        )
    )
    monkeypatch.setattr("gemini_research_mcp.deep._get_healthy_client", lambda: fake_client)

    result = await research_followup(
        previous_interaction_id="quick-interaction-123",
        query="Keep the legacy helper contract.",
    )

    assert result == "Follow-up answer."


@pytest.mark.asyncio
async def test_research_followup_serializes_next_interaction_id() -> None:
    """The MCP result should expose the latest interaction for another turn."""
    from gemini_research_mcp.server import research_followup

    with patch(
        "gemini_research_mcp.server._research_followup",
        new_callable=AsyncMock,
        return_value=("Follow-up answer.", "followup-interaction-456"),
    ):
        response = await research_followup(
            query="What changed?",
            interaction_id="quick-interaction-123",
        )

    assert "Interaction ID: `followup-interaction-456`" in response


@pytest.mark.asyncio
async def test_research_web_serializes_quick_interaction_id() -> None:
    """The MCP result should make the quick interaction usable by research_followup."""
    from gemini_research_mcp.server import research_web

    with patch(
        "gemini_research_mcp.server.quick_research",
        new_callable=AsyncMock,
        return_value=ResearchResult(
            text="Current answer.",
            interaction_id="quick-interaction-123",
        ),
    ):
        response = await research_web(query="Current question")

    assert "Interaction ID: `quick-interaction-123`" in response
