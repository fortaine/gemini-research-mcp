"""
Data types for Gemini Research MCP Server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# =============================================================================
# Error Categories (inspired by DanDaDaDanDan/mcp-gemini)
# =============================================================================


class ErrorCategory(StrEnum):
    """Categorized error types for programmatic handling."""

    AUTH_ERROR = "AUTH_ERROR"  # API key invalid or missing
    RATE_LIMIT = "RATE_LIMIT"  # Quota exceeded, retry later
    CONTENT_BLOCKED = "CONTENT_BLOCKED"  # Content policy violation
    SAFETY_BLOCK = "SAFETY_BLOCK"  # Safety filter triggered
    TIMEOUT = "TIMEOUT"  # Research exceeded max time
    NOT_FOUND = "NOT_FOUND"  # Interaction ID not found
    RESEARCH_FAILED = "RESEARCH_FAILED"  # Research task failed
    RESEARCH_CANCELLED = "RESEARCH_CANCELLED"  # Research was cancelled
    INTERNAL_ERROR = "INTERNAL_ERROR"  # Unexpected internal error
    API_ERROR = "API_ERROR"  # Other API errors


class DeepResearchAgent(StrEnum):
    """Supported agent for deep research."""

    DEEP_RESEARCH = "deep-research-preview-04-2026"
    DEEP_RESEARCH_MAX = "deep-research-max-preview-04-2026"
    DEEP_RESEARCH_PRO = "deep-research-pro-preview-12-2025"


def _categorize_error_message(message: str) -> ErrorCategory:
    """Infer error category from error message."""
    msg_lower = message.lower()

    if "api key" in msg_lower or "unauthorized" in msg_lower or "401" in msg_lower:
        return ErrorCategory.AUTH_ERROR
    if "rate" in msg_lower or "quota" in msg_lower or "429" in msg_lower:
        return ErrorCategory.RATE_LIMIT
    if "safety" in msg_lower:
        return ErrorCategory.SAFETY_BLOCK
    if "blocked" in msg_lower or "content policy" in msg_lower:
        return ErrorCategory.CONTENT_BLOCKED
    if "timeout" in msg_lower or "timed out" in msg_lower:
        return ErrorCategory.TIMEOUT
    if "not found" in msg_lower or "404" in msg_lower:
        return ErrorCategory.NOT_FOUND
    if "cancelled" in msg_lower or "canceled" in msg_lower:
        return ErrorCategory.RESEARCH_CANCELLED
    if "failed" in msg_lower:
        return ErrorCategory.RESEARCH_FAILED

    return ErrorCategory.API_ERROR


# =============================================================================
# Exceptions
# =============================================================================


class DeepResearchError(Exception):
    """Base error for Deep Research operations.

    Provides structured error information with error codes for programmatic handling.
    """

    def __init__(
        self,
        code: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        category: ErrorCategory | None = None,
    ):
        """Initialize a structured Deep Research error.

        `message` is optional for compatibility with runtime/task wrappers that may
        only provide an error code. We normalize to a deterministic fallback message
        so error construction itself never masks the original failure path.
        """
        resolved_message = (message or "").strip()
        if not resolved_message and details:
            details_message = details.get("message")
            if isinstance(details_message, str) and details_message.strip():
                resolved_message = details_message.strip()
        if not resolved_message:
            resolved_message = "No error message provided."

        self.code = code
        self.message = resolved_message
        self.details = details or {}
        # Auto-categorize if not provided
        self.category = category or _categorize_error_message(resolved_message)
        super().__init__(f"{code}: {resolved_message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "category": self.category.value if self.category else None,
            "message": self.message,
            "details": self.details,
        }

    @property
    def is_retryable(self) -> bool:
        """Whether this error might succeed on retry."""
        return self.category in (
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.API_ERROR,
        )


# =============================================================================
# Source Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class Source:
    """A source/citation from grounded search."""

    uri: str
    title: str


@dataclass(slots=True)
class ParsedCitation:
    """A citation extracted from the report with resolved URL.

    Deep Research reports include citations with vertexaisearch redirect URLs.
    This class stores both the original redirect and the resolved real URL.
    """

    number: int
    domain: str
    url: str | None = None
    title: str | None = None
    redirect_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "number": self.number,
            "domain": self.domain,
            "url": self.url,
            "title": self.title,
            "redirect_url": self.redirect_url,
        }


# =============================================================================
# Usage Tracking
# =============================================================================


@dataclass(slots=True)
class DeepResearchUsage:
    """Token usage and cost information for a Deep Research task."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cost: float | None = None
    completion_cost: float | None = None
    total_cost: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


# =============================================================================
# Research Results
# =============================================================================


@dataclass(slots=True)
class ResearchResult:
    """Result from quick_research()."""

    text: str
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    thinking_summary: str | None = None


@dataclass(slots=True)
class DeepResearchResult:
    """Result from deep_research()."""

    text: str
    text_without_sources: str | None = None
    citations: list[Source] = field(default_factory=list)
    parsed_citations: list[ParsedCitation] = field(default_factory=list)
    thinking_summaries: list[str] = field(default_factory=list)
    interaction_id: str | None = None
    usage: DeepResearchUsage | None = None
    duration_seconds: float | None = None
    raw_interaction: Any = None
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        citations = [c.to_dict() for c in self.parsed_citations] if self.parsed_citations else []
        return {
            "id": self.interaction_id,
            "text": self.text,
            "text_without_sources": self.text_without_sources,
            "citations": citations,
            "thinking_summaries": self.thinking_summaries,
            "usage": self.usage.to_dict() if self.usage else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class DeepResearchProgress:
    """Progress update from streaming deep research."""

    # "start", "thought", "text", "image", "action", "complete", "plan_ready",
    # "error", "status". "image" is distinct from "text" so image content/deltas
    # are never concatenated into the report text. "plan_ready" is emitted when
    # collaborative_planning=True and the agent is waiting for plan approval
    # (Interactions API status "requires_action").
    event_type: str
    content: str | None = None  # For "image": base64-encoded data, when provided inline.
    interaction_id: str | None = None
    event_id: str | None = None  # For stream resumption after disconnection
    image_mime_type: str | None = None  # For "image" events: e.g. "image/png".
    image_uri: str | None = None  # For "image" events: hosted URI, when data isn't inlined.


# =============================================================================
# File Search Store (RAG)
# TODO: These types are defined for future use with file search capabilities
# =============================================================================


@dataclass(frozen=True, slots=True)
class FileSearchStore:
    """A file search store for RAG (future use)."""

    name: str
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class FileSearchDocument:
    """A document in a file search store (future use)."""

    name: str
    display_name: str | None = None
    file_name: str | None = None
