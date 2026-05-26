"""Unit tests for data types.

Tests dataclass construction, serialization, and error handling.
Run with: uv run pytest tests/test_types.py -v
"""

import pytest

from gemini_research_mcp.types import (
    DeepResearchError,
    DeepResearchProgress,
    DeepResearchResult,
    DeepResearchUsage,
    ParsedCitation,
    ResearchResult,
    Source,
)


class TestSource:
    """Test Source dataclass."""

    def test_construction(self):
        """Source should be constructable with uri and title."""
        source = Source(uri="https://example.com", title="Example")
        assert source.uri == "https://example.com"
        assert source.title == "Example"

    def test_frozen(self):
        """Source should be immutable (frozen)."""
        source = Source(uri="https://example.com", title="Example")
        with pytest.raises(AttributeError):
            source.uri = "https://other.com"  # type: ignore[misc]

    def test_slots(self):
        """Source should use slots for memory efficiency."""
        source = Source(uri="https://example.com", title="Example")
        assert not hasattr(source, "__dict__")


class TestParsedCitation:
    """Test ParsedCitation dataclass."""

    def test_construction_minimal(self):
        """ParsedCitation should work with minimal args."""
        citation = ParsedCitation(number=1, domain="example.com")
        assert citation.number == 1
        assert citation.domain == "example.com"
        assert citation.url is None
        assert citation.title is None
        assert citation.redirect_url is None

    def test_construction_full(self):
        """ParsedCitation should work with all args."""
        citation = ParsedCitation(
            number=1,
            domain="example.com",
            url="https://example.com/page",
            title="Example Page",
            redirect_url="https://vertexaisearch.cloud.google.com/...",
        )
        assert citation.number == 1
        assert citation.url == "https://example.com/page"
        assert citation.title == "Example Page"

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        citation = ParsedCitation(
            number=1,
            domain="example.com",
            url="https://example.com",
            title="Example",
            redirect_url="https://redirect.com",
        )
        d = citation.to_dict()
        assert d["number"] == 1
        assert d["domain"] == "example.com"
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"
        assert d["redirect_url"] == "https://redirect.com"


class TestDeepResearchUsage:
    """Test DeepResearchUsage dataclass."""

    def test_construction_empty(self):
        """DeepResearchUsage should work with no args."""
        usage = DeepResearchUsage()
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.total_tokens is None
        assert usage.raw_usage == {}

    def test_construction_full(self):
        """DeepResearchUsage should work with all args."""
        usage = DeepResearchUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_cost=0.001,
            completion_cost=0.002,
            total_cost=0.003,
            raw_usage={"key": "value"},
        )
        assert usage.prompt_tokens == 1000
        assert usage.total_cost == 0.003

    def test_to_dict(self):
        """to_dict should serialize key fields."""
        usage = DeepResearchUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            total_cost=0.01,
        )
        d = usage.to_dict()
        assert d["prompt_tokens"] == 1000
        assert d["completion_tokens"] == 500
        assert d["total_tokens"] == 1500
        assert d["total_cost"] == 0.01


class TestResearchResult:
    """Test ResearchResult dataclass."""

    def test_construction_minimal(self):
        """ResearchResult should work with just text."""
        result = ResearchResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.sources == []
        assert result.queries == []
        assert result.thinking_summary is None

    def test_construction_full(self):
        """ResearchResult should work with all args."""
        sources = [Source(uri="https://example.com", title="Example")]
        result = ResearchResult(
            text="Research findings",
            sources=sources,
            queries=["query 1", "query 2"],
            thinking_summary="I thought about it",
        )
        assert len(result.sources) == 1
        assert len(result.queries) == 2
        assert result.thinking_summary == "I thought about it"


class TestDeepResearchResult:
    """Test DeepResearchResult dataclass."""

    def test_construction_minimal(self):
        """DeepResearchResult should work with just text."""
        result = DeepResearchResult(text="Deep research report")
        assert result.text == "Deep research report"
        assert result.citations == []
        assert result.parsed_citations == []
        assert result.thinking_summaries == []
        assert result.interaction_id is None
        assert result.usage is None

    def test_to_dict(self):
        """to_dict should serialize complex result."""
        citations = [
            ParsedCitation(number=1, domain="example.com", url="https://example.com")
        ]
        usage = DeepResearchUsage(total_tokens=1000, total_cost=0.01)
        result = DeepResearchResult(
            text="# Report\n\nFindings...",
            text_without_sources="# Report\n\nFindings...",
            parsed_citations=citations,
            thinking_summaries=["Thought 1", "Thought 2"],
            interaction_id="int_123",
            usage=usage,
            duration_seconds=120.5,
        )
        d = result.to_dict()
        assert d["id"] == "int_123"
        assert d["text"] == "# Report\n\nFindings..."
        assert len(d["citations"]) == 1
        assert d["citations"][0]["domain"] == "example.com"
        assert d["thinking_summaries"] == ["Thought 1", "Thought 2"]
        assert d["usage"]["total_tokens"] == 1000
        assert d["duration_seconds"] == 120.5

    def test_to_dict_empty_citations(self):
        """to_dict should handle empty citations."""
        result = DeepResearchResult(text="Report")
        d = result.to_dict()
        assert d["citations"] == []

    def test_to_dict_no_usage(self):
        """to_dict should handle None usage."""
        result = DeepResearchResult(text="Report")
        d = result.to_dict()
        assert d["usage"] is None


class TestDeepResearchProgress:
    """Test DeepResearchProgress dataclass."""

    def test_construction_minimal(self):
        """DeepResearchProgress should work with just event_type."""
        progress = DeepResearchProgress(event_type="start")
        assert progress.event_type == "start"
        assert progress.content is None
        assert progress.interaction_id is None
        assert progress.event_id is None

    def test_construction_full(self):
        """DeepResearchProgress should work with all args."""
        progress = DeepResearchProgress(
            event_type="thought",
            content="Analyzing the query...",
            interaction_id="int_123",
            event_id="evt_456",
        )
        assert progress.event_type == "thought"
        assert progress.content == "Analyzing the query..."
        assert progress.interaction_id == "int_123"
        assert progress.event_id == "evt_456"

    def test_event_types(self):
        """Should support all expected event types."""
        for event_type in ["start", "thought", "text", "complete", "error", "status"]:
            progress = DeepResearchProgress(event_type=event_type)
            assert progress.event_type == event_type

    def test_event_id_for_resumption(self):
        """event_id should be usable for stream resumption."""
        progress = DeepResearchProgress(
            event_type="thought",
            event_id="evt_abc123",
        )
        assert progress.event_id == "evt_abc123"


class TestDeepResearchError:
    """Test DeepResearchError exception."""

    def test_construction(self):
        """DeepResearchError should be constructable."""
        error = DeepResearchError(
            code="TIMEOUT",
            message="Research timed out after 20 minutes",
        )
        assert error.code == "TIMEOUT"
        assert error.message == "Research timed out after 20 minutes"
        assert error.details == {}

    def test_construction_without_message_uses_fallback(self):
        """Missing message should not raise and should use deterministic fallback."""
        error = DeepResearchError(code="INTERNAL_ERROR")
        assert error.code == "INTERNAL_ERROR"
        assert error.message == "No error message provided."
        assert str(error) == "INTERNAL_ERROR: No error message provided."

    def test_construction_uses_message_from_details_when_missing(self):
        """details['message'] should be used when explicit message is missing."""
        error = DeepResearchError(
            code="API_ERROR",
            details={"message": "Provider returned malformed payload"},
        )
        assert error.message == "Provider returned malformed payload"

    def test_construction_with_details(self):
        """DeepResearchError should accept details."""
        error = DeepResearchError(
            code="API_ERROR",
            message="API returned 500",
            details={"status_code": 500, "response": "Internal Server Error"},
        )
        assert error.details["status_code"] == 500

    def test_str_representation(self):
        """DeepResearchError should have readable str."""
        error = DeepResearchError(code="TEST", message="Test message")
        assert str(error) == "TEST: Test message"

    def test_to_dict(self):
        """to_dict should serialize error."""
        error = DeepResearchError(
            code="RESEARCH_FAILED",
            message="Failed to complete",
            details={"interaction_id": "int_123"},
        )
        d = error.to_dict()
        assert d["code"] == "RESEARCH_FAILED"
        assert d["message"] == "Failed to complete"
        assert d["details"]["interaction_id"] == "int_123"

    def test_is_exception(self):
        """DeepResearchError should be raisable."""
        with pytest.raises(DeepResearchError) as exc_info:
            raise DeepResearchError(code="TEST", message="Test error")
        assert exc_info.value.code == "TEST"
