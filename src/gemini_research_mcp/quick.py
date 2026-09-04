"""
Quick research using Gemini grounded search.

Provides fast web research with citations in 5-30 seconds.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    ThinkingConfig,
    ThinkingLevel,
    Tool,
)
from pydantic import BaseModel, Field

from gemini_research_mcp.config import (
    LOGGER_NAME,
    default_system_prompt,
    get_api_key,
    get_model,
    get_summary_model,
)
from gemini_research_mcp.types import ResearchResult, Source

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse

logger = logging.getLogger(LOGGER_NAME)

# Map string levels to ThinkingLevel enum
THINKING_LEVEL_MAP = {
    "minimal": ThinkingLevel.MINIMAL,
    "low": ThinkingLevel.LOW,
    "medium": ThinkingLevel.MEDIUM,
    "high": ThinkingLevel.HIGH,
}


def _get_thinking_level(level: str) -> ThinkingLevel:
    """Convert string level to ThinkingLevel enum."""
    return THINKING_LEVEL_MAP.get(level.lower(), ThinkingLevel.HIGH)


def _extract_sources_from_text(text: str) -> list[Source]:
    """Extract markdown or bare URLs from response text when grounding metadata is absent."""
    sources: list[Source] = []
    seen_uris: set[str] = set()

    for title, uri in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", text):
        cleaned_uri = uri.strip()
        if cleaned_uri in seen_uris:
            continue
        seen_uris.add(cleaned_uri)
        sources.append(Source(uri=cleaned_uri, title=title.strip() or cleaned_uri))

    if sources:
        return sources

    for uri in re.findall(r"https?://[^\s)>\]]+", text):
        cleaned_uri = uri.rstrip(".,)")
        if cleaned_uri in seen_uris:
            continue
        seen_uris.add(cleaned_uri)
        hostname = urlparse(cleaned_uri).netloc or cleaned_uri
        sources.append(Source(uri=cleaned_uri, title=hostname))

    return sources


def _extract_sources(response: GenerateContentResponse) -> tuple[list[Source], list[str]]:
    """Extract sources and queries from grounding metadata.

    Falls back to parsing URLs out of the response text whenever grounding
    metadata is absent or empty - e.g. when the model answers from its own
    knowledge without invoking the google_search tool at all.
    """
    sources: list[Source] = []
    queries: list[str] = []

    candidate = response.candidates[0] if response.candidates else None
    gm = candidate.grounding_metadata if candidate else None

    if gm:
        # Extract search queries used
        if gm.web_search_queries:
            queries = list(gm.web_search_queries)

        # Extract sources from grounding chunks
        if gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    sources.append(
                        Source(
                            uri=chunk.web.uri or "",
                            title=chunk.web.title or "",
                        )
                    )

    if not sources and response.text:
        sources = _extract_sources_from_text(response.text)

    return sources, queries


async def quick_research(
    query: str,
    *,
    model: str | None = None,
    thinking_level: str | None = None,
    system_instruction: str | None = None,
    include_thoughts: bool = False,
) -> ResearchResult:
    """
    Fast grounded search using google_search tool.

    Returns response grounded in real-time web search results.
    Typically completes in 5-30 seconds.

    Args:
        query: Research question or topic
        model: Gemini model (default: configured GEMINI_MODEL / gemini-3.8-flash)
        thinking_level: Accepted for backward compatibility but ignored; quick research
            always uses high thinking
        system_instruction: Optional system prompt
        include_thoughts: If True, include thinking summary in result

    Returns:
        ResearchResult with text, sources, queries, and optional thinking summary
    """
    client = genai.Client(api_key=get_api_key())
    model = model or get_model()

    if thinking_level and thinking_level.lower() != "high":
        logger.info(
            "Ignoring requested thinking level '%s'; quick_research uses fixed high thinking",
            thinking_level,
        )

    config = GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())],
        thinking_config=ThinkingConfig(
            thinking_level=ThinkingLevel.HIGH,
            include_thoughts=include_thoughts,
        ),
        system_instruction=system_instruction or default_system_prompt(),
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=query,
        config=config,
    )

    sources, queries = _extract_sources(response)

    # Extract thinking summary if present
    thinking_summary = None
    if include_thoughts and response.candidates:
        content = response.candidates[0].content
        if content is not None and content.parts is not None:
            for part in content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_summary = part.text
                    break

    return ResearchResult(
        text=response.text or "",
        sources=sources,
        queries=queries,
        thinking_summary=thinking_summary,
    )


# =============================================================================
# Metadata Generation (Structured Output)
# =============================================================================


class SessionMetadata(BaseModel):
    """Generated metadata for a research session."""

    title: str = Field(description="Short descriptive title for the research report")
    summary: str = Field(description="Concise 2-3 sentence summary of key findings")


async def generate_session_metadata(
    text: str,
    query: str,
    *,
    max_title_chars: int = 60,
    max_summary_chars: int = 300,
) -> SessionMetadata:
    """
    Generate title and summary for a research session in a single API call.

    Uses Gemini Flash with Pydantic structured output for efficiency.
    Cost: ~$0.0003 per call (~100 output tokens, shared input).

    Args:
        text: The full research report text
        query: The original research query (for context)
        max_title_chars: Maximum characters for title (default 60)
        max_summary_chars: Maximum characters for summary (default 300)

    Returns:
        SessionMetadata with title and summary
    """
    if not text:
        return SessionMetadata(title="", summary="")

    client = genai.Client(api_key=get_api_key())
    model = get_summary_model()

    # Truncate input to first ~2000 chars to minimize tokens
    input_text = text[:2000]
    if len(text) > 2000:
        input_text += "..."

    prompt = f"""Analyze this research report and generate metadata.

Original query: {query}

Report:
{input_text}

Generate:
1. A short, descriptive title (max {max_title_chars} chars) - capture the main topic, 
   suitable for a document title. No quotes. No "Research on" or "Analysis of" prefixes.
2. A concise summary (max {max_summary_chars} chars) - 2-3 sentences covering key findings."""

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_level=ThinkingLevel.LOW,  # low thinking keeps helper calls concise
        ),
        response_mime_type="application/json",
        response_schema=SessionMetadata,
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        # Use response.parsed for automatic Pydantic parsing
        metadata: SessionMetadata = response.parsed  # type: ignore[assignment]

        # Ensure we stay within limits
        title = metadata.title.strip().strip('"\'')
        summary = metadata.summary.strip()

        if len(title) > max_title_chars:
            title = title[: max_title_chars - 3] + "..."
        if len(summary) > max_summary_chars:
            summary = summary[: max_summary_chars - 3] + "..."

        return SessionMetadata(title=title, summary=summary)
    except Exception as e:
        logger.warning("Metadata generation failed: %s", e)
        return SessionMetadata(title="", summary="")


class TitleOnly(BaseModel):
    """Structured output for title generation from query."""

    title: str = Field(description="Short descriptive title for the research topic")


async def generate_title_from_query(
    query: str,
    *,
    max_chars: int = 60,
) -> str:
    """
    Generate a descriptive title from just the research query.

    Uses Gemini Flash with minimal thinking for fast generation.
    Cost: ~$0.0001 per call (very short input/output).

    This is used at the START of research before we have results,
    so we generate a title based on what the user is asking about.

    Args:
        query: The research query
        max_chars: Maximum characters for title (default 60)

    Returns:
        A descriptive title, or empty string on failure
    """
    if not query:
        return ""

    client = genai.Client(api_key=get_api_key())
    model = get_summary_model()

    # Truncate query if very long
    input_query = query[:500] if len(query) > 500 else query

    prompt = f"""Generate a short, descriptive title for this research request.

Research request: {input_query}

Generate a title (max {max_chars} chars) that captures the main topic.
- No quotes
- No "Research on" or "Analysis of" prefixes
- Clear and concise, suitable for a document title"""

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_level=ThinkingLevel.LOW,  # low thinking keeps helper calls concise
        ),
        response_mime_type="application/json",
        response_schema=TitleOnly,
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        result: TitleOnly = response.parsed  # type: ignore[assignment]
        title = result.title.strip().strip('"\'')

        if len(title) > max_chars:
            title = title[: max_chars - 3] + "..."

        return title
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return ""


async def semantic_match_session(
    user_query: str,
    sessions: list[dict[str, str]],
) -> str | None:
    """
    Find the research session that best matches a user's follow-up question.

    Uses Gemini 3.0 Flash with minimal thinking for fast semantic matching.
    Cost: ~$0.0003 per call.

    Args:
        user_query: The user's follow-up question
        sessions: List of dicts with 'id', 'query', and 'summary' keys

    Returns:
        The interaction_id of the best matching session, or None if no match
    """
    if not sessions:
        return None

    if len(sessions) == 1:
        # Only one session - return it directly
        return sessions[0]["id"]

    client = genai.Client(api_key=get_api_key())
    model = get_summary_model()

    # Build session list for prompt (truncate summaries to avoid context overflow)
    session_list = []
    for i, s in enumerate(sessions, 1):
        summary = s.get('summary', 'No summary')[:300]  # Limit summary length
        session_list.append(
            f"{i}. [{s['id']}] \"{s['query'][:100]}\"\n   Summary: {summary}"
        )

    prompt = f"""Given these research sessions and a user's follow-up question,
which session is the user most likely referring to?

Research Sessions:
{chr(10).join(session_list)}

User's follow-up question: "{user_query}"

Return ONLY the interaction_id (the value in brackets) of the best matching session.
If none of the sessions match the user's question, return exactly: NONE"""

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_level=ThinkingLevel.LOW,  # low thinking keeps helper calls concise
        ),
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        result = (response.text or "").strip()

        # Handle no match
        if result.upper() == "NONE":
            return None

        # Validate the returned ID exists in our sessions
        valid_ids = {s["id"] for s in sessions}
        if result in valid_ids:
            return result

        # Try to extract ID if model added extra text
        for session_id in valid_ids:
            if session_id in result:
                return session_id

        logger.warning("semantic_match_session: invalid response '%s'", result)
        return None

    except Exception as e:
        logger.warning("semantic_match_session failed: %s", e)
        return None
