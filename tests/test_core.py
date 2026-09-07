"""Unit tests for core research functions.

Tests thinking level parsing and system prompt generation.
Run with: uv run pytest tests/ -v
"""

import types
from datetime import date

import pytest
from google.genai.types import ThinkingLevel

from gemini_research_mcp.config import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    default_system_prompt,
)
from gemini_research_mcp.quick import (
    THINKING_LEVEL_MAP,
    _extract_sources_from_text,
    _get_thinking_level,
    quick_research,
)


class TestThinkingLevel:
    """Test thinking level parsing for Gemini 3 models."""

    @pytest.mark.parametrize("level,expected", [
        ("minimal", ThinkingLevel.MINIMAL),
        ("low", ThinkingLevel.LOW),
        ("medium", ThinkingLevel.MEDIUM),
        ("high", ThinkingLevel.HIGH),
    ])
    def test_named_levels(self, level: str, expected: ThinkingLevel):
        """Named levels should map to correct ThinkingLevel enum."""
        assert _get_thinking_level(level) == expected

    def test_case_insensitive(self):
        """Level names should be case-insensitive."""
        assert _get_thinking_level("HIGH") == ThinkingLevel.HIGH
        assert _get_thinking_level("High") == ThinkingLevel.HIGH

    def test_unknown_level_defaults_to_high(self):
        """Unknown level names should default to HIGH."""
        assert _get_thinking_level("unknown") == ThinkingLevel.HIGH
        assert _get_thinking_level("max") == ThinkingLevel.HIGH  # 'max' not valid for Gemini 3

    def test_level_map_complete(self):
        """THINKING_LEVEL_MAP should have all expected levels."""
        expected = {"minimal", "low", "medium", "high"}
        assert set(THINKING_LEVEL_MAP.keys()) == expected

    def test_source_fallback_extracts_markdown_links(self):
        """Link extraction should populate sources when grounding metadata is unavailable."""
        sources = _extract_sources_from_text(
            "See the [Python docs](https://docs.python.org/3/library/dataclasses.html)."
        )

        assert len(sources) == 1
        assert sources[0].title == "Python docs"
        assert sources[0].uri == "https://docs.python.org/3/library/dataclasses.html"

    @pytest.mark.asyncio
    async def test_quick_research_uses_interactions_with_fixed_high_thinking(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """quick_research should create a stateful interaction with high thinking."""
        captured: dict[str, object] = {}

        async def fake_create(**kwargs: object) -> object:
            captured.update(kwargs)
            return types.SimpleNamespace(id="interaction-123", output_text="ok", steps=[])

        fake_client = types.SimpleNamespace(
            aio=types.SimpleNamespace(
                interactions=types.SimpleNamespace(create=fake_create)
            )
        )

        monkeypatch.setattr("gemini_research_mcp.quick.get_api_key", lambda: "test-key")
        monkeypatch.setattr("gemini_research_mcp.quick.get_model", lambda: "gemini-3.1-pro-preview")
        monkeypatch.setattr(
            "gemini_research_mcp.quick.genai.Client",
            lambda api_key: fake_client,
        )

        await quick_research("test query", thinking_level="minimal")

        assert captured["generation_config"] == {"thinking_level": "high"}
        assert captured["tools"] == [{"type": "google_search"}]

    def test_default_is_high(self):
        """Default thinking level should be 'high'."""
        assert DEFAULT_THINKING_LEVEL == "high"


class TestSystemPrompt:
    """Test default system prompt generation."""

    def test_includes_current_date(self):
        """System prompt should include current date."""
        prompt = default_system_prompt()
        today = date.today().strftime("%B %d, %Y")
        assert today in prompt

    def test_mentions_research_role(self):
        """System prompt should establish research analyst role."""
        prompt = default_system_prompt()
        assert "research" in prompt.lower()
        assert "analyst" in prompt.lower() or "expert" in prompt.lower()

    def test_mentions_citations(self):
        """System prompt should mention citing sources."""
        prompt = default_system_prompt()
        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_reasonable_length(self):
        """System prompt should be reasonably sized."""
        prompt = default_system_prompt()
        assert 200 < len(prompt) < 2000, f"Prompt length {len(prompt)} seems unusual"


class TestConstants:
    """Test module constants."""

    def test_default_model_is_gemini(self):
        """Default model should be a Gemini model."""
        assert "gemini" in DEFAULT_MODEL.lower()

    def test_default_thinking_level_is_high(self):
        """Default thinking level should be 'high' for quality results."""
        assert DEFAULT_THINKING_LEVEL == "high"
