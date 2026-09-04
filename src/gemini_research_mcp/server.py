"""
Gemini Research MCP Server

Provides AI-powered research tools via Gemini:
- research_web: Fast grounded web search (5-30 seconds) - Gemini + Google Search
- research_deep: Comprehensive multi-step research (3-20 minutes) - Deep Research Agent
- research_followup: Ask follow-up questions about completed research

Architecture:
- FastMCP 4 with TasksExtension-based task support for background tasks (MCP Tasks / SEP-1732)
- Task routing via TaskConfig(mode="required") with in-memory Docket backend
- Sessionless guard-pattern elicitation with legacy Context.elicit() compatibility
- Progress reporting via ctx.report_progress() (task statusMessage channel)
"""

# NOTE: Do NOT use `from __future__ import annotations` with FastMCP/Pydantic
# as it breaks type resolution for Annotated parameters in tool functions

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.server.transforms.search import BM25SearchTransform
from fastmcp.utilities.tasks import TaskConfig
from fastmcp_tasks import TasksExtension

# Raw MCP types
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    Icon,
    InputRequiredResult,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import BaseModel, Field

from gemini_research_mcp import __version__
from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import (
    LOGGER_NAME,
    get_deep_research_agent,
    get_export_dir,
    get_model,
    get_tasks_backend_url,
    is_retryable_error,
)
from gemini_research_mcp.content import fetch_webpage as _fetch_webpage
from gemini_research_mcp.deep import (
    deep_research_stream,
    get_research_status,
    validate_mcp_servers_supported,
)
from gemini_research_mcp.deep import (
    inspect_mcp_server_for_gemini as _inspect_mcp_server_for_gemini,
)
from gemini_research_mcp.deep import (
    research_followup as _research_followup,
)
from gemini_research_mcp.export import (
    ExportFormat,
    ExportResult,
    export_session,
)
from gemini_research_mcp.quick import (
    generate_session_metadata,
    generate_title_from_query,
    quick_research,
    semantic_match_session,
)
from gemini_research_mcp.storage import (
    ExportArtifact,
    ResearchSession,
    ResearchStatus,
    delete_research_session,
    get_export_store,
    get_research_session,
    list_resumable_sessions,
    save_research_session,
    update_research_session,
)
from gemini_research_mcp.storage import (
    list_research_sessions as _list_sessions,
)
from gemini_research_mcp.types import (
    DeepResearchAgent,
    DeepResearchError,
    DeepResearchResult,
    ErrorCategory,
)

# Configure logging
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Gemini icon URL (VS Code doesn't support SVG icons)
# See: https://github.com/microsoft/vscode/issues/290809
GEMINI_ICON_URL = "https://raw.githubusercontent.com/machinemates-ai/gemini-research-mcp/main/vscode-extension/icon.png"


# =============================================================================
# Export Artifact Storage
# =============================================================================

# TTL for exported files (1 hour) - enforced by the backend (disk cache TTL or
# Redis EXPIRE), not by application-level bookkeeping. See storage.py's
# ExportArtifactStore, which replaced the old in-memory `_export_cache` dict so
# exports survive restarts and can be shared across worker processes when
# GEMINI_RESEARCH_STORAGE_URL points multiple instances at the same backend.
EXPORT_TTL_SECONDS = 3600
STALE_RESUMABLE_SECONDS = 24 * 60 * 60
RECENT_FAILED_SECONDS = 24 * 60 * 60
DEEP_RESEARCH_POLL_MAX_WAIT_SECONDS = 1200
DEEP_RESEARCH_POLL_INTERVAL_SECONDS = 10


async def _cache_export(result: ExportResult, session_id: str) -> str:
    """Persist an export artifact and return its unique ID."""
    return await get_export_store().save_async(
        session_id=session_id,
        filename=result.filename,
        format=result.format.value,
        mime_type=result.mime_type,
        content=result.content,
    )


async def _get_cached_export(export_id: str) -> ExportArtifact | None:
    """Retrieve a persisted export artifact, or None if expired/missing."""
    return await get_export_store().get_async(export_id)


# =============================================================================
# Server Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Check for resumable sessions on startup.

    Task support is registered through FastMCP 4 TasksExtension, which
    discovers TaskConfig-enabled tools and installs the task protocol handlers.
    """
    logger.info("✅ FastMCP Docket task support active")

    # Check for resumable sessions from previous runs
    try:
        resumable = list_resumable_sessions(limit=10)
        if resumable:
            logger.info(
                "🔄 %d resumable session(s) — use resume_research to recover",
                len(resumable),
            )
    except Exception as e:
        logger.warning("Failed to check for resumable sessions: %s", e)

    yield


# =============================================================================
# Server Instance
# =============================================================================

mcp = FastMCP(
    name="Gemini Research",
    version=__version__,
    icons=[Icon(src=GEMINI_ICON_URL, mime_type="image/png")],
    instructions="""
Gemini Research MCP Server - AI-powered research toolkit

## Always Visible Tools

### Quick Lookup (research_web)
Fast web research with Gemini grounding (5-30 seconds).
Use for: fact-checking, current events, documentation, "what is", "how to".

### Deep Research (research_deep)
Comprehensive autonomous research agent (3-20 minutes). **Requires MCP Tasks
support on the client** (SEP-1732). Clients without Tasks capability will
receive a `-32600` error — upgrade the client or use `research_web`.
Use for: research reports, competitive analysis, "compare", "analyze", "investigate".
- Automatically asks clarifying questions for vague queries
- Streams progress via the Tasks channel
- Returns a comprehensive report with citations
- Sessions are persisted at the START so they can be recovered with
  `resume_research` if the client disconnects mid-run

### Resume Research (resume_research)
Resume an interrupted or in-progress `research_deep` run. Also the way to
list recoverable sessions when called without arguments.

### Export Research (export_research_session)
Save a completed session to disk as Word (`.docx`), Markdown, or JSON.
**Writes to disk by default** (under `GEMINI_RESEARCH_EXPORT_DIR`, default
`~/.gemini-research/exports/`); the absolute path is returned in the
response. GUI clients additionally receive an embedded resource for
"Save As".

## Discover Utility Tools With search_tools
This server uses BM25 tool search to keep the visible tool list small.
Use `search_tools` with a natural-language query to discover utility tools for:
- reading and extracting webpage content
- browsing reusable report templates
- listing saved research sessions
- asking follow-up questions on prior research

Use `call_tool` to invoke a discovered tool by name with arguments.
Clients that already know a hidden tool name can still call it directly.

**Workflow:**
- Simple questions → `research_web`
- Complex questions → `research_deep` (requires Tasks)
- Run was interrupted → `resume_research`
- Ready to hand off a report → `export_research_session(format="docx")`
- Need another utility → `search_tools`, then `call_tool`
""",
    lifespan=lifespan,
)

# FastMCP 4 no longer wires the Docket task backend implicitly: TaskConfig-enabled
# tools require the SEP-2663 tasks extension to be registered explicitly.
# FASTMCP_DOCKET_URL explicitly configures the backend. Otherwise a configured
# shared Redis storage URL is reused; None preserves the local memory default.
mcp.add_extension(
    TasksExtension(url=get_tasks_backend_url(), name="gemini-research-mcp")
)

# Keep the visible tool catalog compact via BM25 relevance search. The five
# tools most clients need immediately stay always-visible in list_tools();
# the rest (fetch_webpage, research_followup, list_research_sessions,
# list_format_templates, refine_research_plan, inspect_mcp_server_for_gemini)
# are discoverable via search_tools and remain
# directly callable by any client that already knows their name.
mcp.add_transform(
    BM25SearchTransform(
        always_visible=[
            "research_web",
            "research_deep",
            "research_deep_max",
            "resume_research",
            "export_research_session",
        ],
        max_results=5,
    )
)


# =============================================================================
# Helper Functions
# =============================================================================


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _resume_hint(interaction_id: str) -> str:
    """Return a concise recovery hint for a persisted Deep Research interaction."""
    return f"Call `resume_research(interaction_id=\"{interaction_id}\")` to check it."


def _refine_plan_hint(interaction_id: str) -> str:
    """Return a concise hint for refining/approving a collaborative-planning plan."""
    return (
        f"Call `refine_research_plan(previous_interaction_id=\"{interaction_id}\", "
        f'decision="approve")` to run this plan, or '
        f'decision="iterate" with `instructions=...` to request changes.'
    )


# =============================================================================
# Helper Functions - Report Formatting
# =============================================================================


def _format_deep_research_report(
    result: DeepResearchResult,
    interaction_id: str,
    elapsed: float,
    image_uris: list[str] | None = None,
) -> str:
    """Format a deep research result into a markdown report."""
    lines = ["## Research Report"]

    if result.text:
        lines.append(result.text)
    else:
        lines.append("*No report available.*")

    # Images (agent_config.visualization="auto") - resource links, never inlined
    # into the report text itself so they can't be confused with citations.
    if image_uris:
        lines.extend(["", "## Images"])
        for i, uri in enumerate(image_uris, start=1):
            lines.append(f"- [Image {i}]({uri})")

    # Usage stats
    if result.usage:
        lines.extend(["", "## Usage"])
        if result.usage.total_tokens:
            lines.append(f"- Total tokens: {result.usage.total_tokens}")
        if result.usage.total_cost:
            lines.append(f"- Estimated cost: ${result.usage.total_cost:.4f}")

    # Duration
    lines.extend(
        [
            "",
            "---",
            f"- Duration: {_format_duration(elapsed)}",
            f"- Interaction ID: `{interaction_id}`",
        ]
    )

    return "\n".join(lines)


def _format_plan_response(plan_text: str, interaction_id: str) -> str:
    """Format a collaborative-planning "plan_ready" response (not the final report)."""
    lines = [
        "## Research Plan (awaiting approval)",
        "",
        plan_text or "*No plan text was returned by the agent.*",
        "",
        "---",
        f"- Interaction ID: `{interaction_id}`",
        f"- {_refine_plan_hint(interaction_id)}",
    ]
    return "\n".join(lines)


async def _persist_deep_research_image(
    *,
    session_id: str,
    index: int,
    data_b64: str | None,
    mime_type: str | None,
    uri: str | None,
) -> tuple[str | None, str]:
    """Persist a Deep Research image as an export artifact.

    Returns (export_id, resource_uri). When no inline base64 data is available
    (only a hosted `uri`), the image is not re-persisted - the original URI is
    passed through as-is, and export_id is None.
    """
    if not data_b64:
        return None, uri or ""

    import base64
    import mimetypes

    content = base64.b64decode(data_b64)
    resolved_mime = mime_type or "image/png"
    extension = mimetypes.guess_extension(resolved_mime) or ".png"
    filename = f"deep-research-image-{index}{extension}"

    export_id = await get_export_store().save_async(
        session_id=session_id,
        filename=filename,
        format="image",
        mime_type=resolved_mime,
        content=content,
    )
    return export_id, f"research://exports/{export_id}"


async def _persist_deep_research_images(
    *, session_id: str, images: list[dict[str, object]]
) -> tuple[list[str], list[str]]:
    """Persist a batch of Deep Research images, returning (export_ids, resource_uris)."""
    export_ids: list[str] = []
    resource_uris: list[str] = []
    for index, image in enumerate(images):
        data = image.get("data")
        mime_type = image.get("mime_type")
        uri = image.get("uri")
        export_id, resource_uri = await _persist_deep_research_image(
            session_id=session_id,
            index=index,
            data_b64=data if isinstance(data, str) else None,
            mime_type=mime_type if isinstance(mime_type, str) else None,
            uri=uri if isinstance(uri, str) else None,
        )
        if resource_uri:
            resource_uris.append(resource_uri)
        if export_id:
            export_ids.append(export_id)
    return export_ids, resource_uris


# =============================================================================
# Tools
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def research_web(
    query: Annotated[str, "Search query or question to research on the web"],
    include_thoughts: Annotated[bool, "Include thinking summary in response"] = False,
) -> str:
    """
    Fast web research with Gemini grounding. Returns answer with citations in seconds.

    Uses a fixed high thinking level for higher-quality grounded answers.

    Use for: quick lookups, fact-checking, current events, documentation, "what is",
    "how to", real-time information, news, API references, error messages.

    Args:
        query: Search query or question to research
        include_thoughts: Include thinking summary in response

    Returns:
        Research results with sources as markdown text
    """
    logger.info("🔎 research_web: %s", query[:100])
    start = time.time()

    try:
        result = await quick_research(
            query=query,
            include_thoughts=include_thoughts,
        )
        elapsed = time.time() - start
        logger.info("   ✅ Completed in %.1fs", elapsed)

        # Format response
        lines = []

        # Main response
        if result.text:
            lines.append(result.text)

        # Sources section
        if result.sources:
            lines.extend(["", "---", "### Sources"])
            for i, source in enumerate(result.sources, 1):
                title = source.title or source.uri
                lines.append(f"{i}. [{title}]({source.uri})")

        # Search queries used
        if result.queries:
            lines.extend(["", "### Search Queries"])
            for q in result.queries:
                lines.append(f"- {q}")

        # Thinking summary (if requested)
        if result.thinking_summary:
            lines.extend(["", "### Thinking Summary", result.thinking_summary])

        # Metadata
        lines.extend(
            [
                "",
                "---",
                f"*Completed in {_format_duration(elapsed)}*",
            ]
        )

        return "\n".join(lines)

    except Exception as e:
        logger.exception("research_web failed: %s", e)
        return f"❌ Research failed: {e}"


# =============================================================================
# Fetch Webpage Tool (FastMCP 3.0 pattern)
# =============================================================================


@mcp.tool(
    annotations=ToolAnnotations(
        read_only_hint=True, open_world_hint=True, idempotent_hint=True
    )
)
async def fetch_webpage(
    url: Annotated[str, "URL of the webpage to fetch and extract content from"],
    max_length: Annotated[
        int | None,
        "Optional max number of characters to return (for chunked reading)",
    ] = None,
    start_index: Annotated[
        int,
        "Character offset to start reading from (for pagination)",
    ] = 0,
    proxy_url: Annotated[
        str | None,
        "Optional HTTP(S) proxy URL for outbound fetch requests",
    ] = None,
) -> str:
    """
    Fetch and extract content from a webpage as Markdown.

    Uses trafilatura for high-quality content extraction (F1: 0.937).
    Falls back to basic HTML parsing if trafilatura is unavailable.

    Security: Blocks requests to private IPs, localhost, and cloud metadata endpoints
    (SSRF protection).

    Use for: Reading articles, documentation, blog posts, extracting content from
    URLs found in research results.

    Args:
        url: The URL to fetch (must be http/https, no private IPs)
        max_length: Optional character limit for chunked responses
        start_index: Character offset for pagination
        proxy_url: Optional HTTP(S) proxy URL

    Returns:
        Extracted content as Markdown, or error message if fetch failed
    """
    logger.info("🌐 fetch_webpage: %s", url[:100])

    result = await _fetch_webpage(
        url,
        max_length=max_length,
        start_index=start_index,
        proxy_url=proxy_url,
    )

    if result.error:
        return f"❌ Failed to fetch: {result.error}"

    # Format the response
    lines = []

    if result.title:
        lines.append(f"# {result.title}")
        lines.append("")

    lines.append(result.content)

    if result.is_truncated:
        next_start = start_index + len(result.content)
        lines.extend([
            "",
            "---",
            (
                "*Content truncated. "
                f"Showing {len(result.content):,} chars starting at index {start_index:,} "
                f"(total ~{result.total_content_length:,} chars). "
                f"Use start_index={next_start} to continue.*"
            ),
        ])

    if result.word_count:
        lines.extend([
            "",
            "---",
            f"*Extracted {result.word_count:,} words from {result.url}*",
        ])

    return "\n".join(lines)


# =============================================================================
# SEP-2322 / SEP-2577: Elicitation, dual-era
# =============================================================================
#
# Two independent MCP eras need two different mechanisms:
#
# - Handshake-era foreground calls (protocol <= 2025-11-25, no MCP Task) keep a
#   bidirectional session channel open for the duration of a tool call, so the
#   server can push a mid-call request via ctx.elicit() and block for the answer.
# - The 2026-07-28 era is sessionless: every request/response pair is a complete,
#   independent transaction. There is no back-channel to push a question on, and
#   the same is true for ANY call running as an MCP Task (research_deep and
#   research_deep_max always run as tasks per TaskConfig(mode="required")).
#   ctx.elicit() raises ToolError in both of these situations.
#
# The sessionless/task replacement is the SEP-2322 guard pattern: the tool
# returns InputRequiredResult describing what it needs; that ends the current
# call; the client answers and calls the SAME tool again with the same
# arguments plus the answer available via ctx.input_responses. Because the
# heuristics below are a pure function of `query`, no server-side state needs
# to survive between the two calls - the tool simply recomputes the same
# `questions` deterministically and reads the answer out of input_responses.

_CLARIFY_INPUT_KEY = "clarify_query"


class ClarificationSchema(BaseModel):
    """Schema for clarification question answers."""

    answer_1: str = Field(default="", description="Answer to first clarifying question")
    answer_2: str = Field(default="", description="Answer to second clarifying question")
    answer_3: str = Field(default="", description="Answer to third clarifying question")


def _uses_sessionless_guard_pattern(ctx: Context) -> bool:
    """True when ctx.elicit() would raise: sessionless protocol era or an MCP Task.

    Mirrors FastMCP's own era check (Context._is_modern_protocol) plus the
    background-task check documented on Context.elicit(): "Imperative
    elicitation is not available inside a background task."
    """
    if ctx.is_background_task:
        return True
    request_context = ctx.request_context
    if request_context is None:
        return False
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS

    return request_context.protocol_version in MODERN_PROTOCOL_VERSIONS


def _build_clarification_guard(query: str, questions: list[str]) -> InputRequiredResult:
    """Build the InputRequiredResult that asks the client for clarification answers."""
    from fastmcp.server.elicitation import get_elicitation_schema
    from mcp.types import ElicitRequest, ElicitRequestFormParams
    from pydantic import create_model

    field_definitions = {
        f"answer_{i + 1}": (str, Field(default="", description=q))
        for i, q in enumerate(questions)
    }
    dynamic_schema = create_model("ClarificationQuestions", **field_definitions)  # type: ignore

    message = (
        f"To improve research quality for:\n\n**\"{query}\"**\n\n"
        f"Please answer these questions (optional - leave blank to continue):"
    )

    request = ElicitRequest(
        params=ElicitRequestFormParams(
            mode="form",
            message=message,
            requested_schema=get_elicitation_schema(dynamic_schema),
        )
    )
    return InputRequiredResult(
        input_requests={_CLARIFY_INPUT_KEY: request},
        # Compact, non-sensitive correlation marker only - never the full
        # request or an authentication secret. The questions themselves are
        # recomputed deterministically from `query` on the resumed call.
        request_state="clarify_query",
    )


def _apply_clarification_answer(query: str, questions: list[str], elicit_result: object) -> str:
    """Fold an answered guard-pattern elicitation back into the query."""
    action = getattr(elicit_result, "action", None)
    content = getattr(elicit_result, "content", None)
    if action != "accept" or not content:
        logger.info("   ⏭️ User skipped/declined/cancelled clarification (guard pattern)")
        return query

    answers = [content.get(f"answer_{i + 1}", "") for i in range(len(questions))]
    non_empty = [a for a in answers if a.strip()]
    if not non_empty:
        logger.info("   ⏭️ User submitted but answers empty (guard pattern)")
        return query

    logger.info("   ✨ User provided %d/%d answers (guard pattern)", len(non_empty), len(questions))
    clarification = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(questions, answers, strict=False) if a.strip()
    )
    refined = f"{query}\n\nAdditional context:\n{clarification}"
    logger.info("   📝 Refined query: %s", refined[:100])
    return refined


def _detect_clarifying_questions(query: str) -> list[str]:
    """Pure heuristic: return clarifying questions for a vague query, or [] if specific enough.

    Deliberately side-effect free so both elicitation eras (and the resumed
    guard-pattern call) recompute the identical question set from `query` alone.
    """
    query_lower = query.lower()
    query_len = len(query)
    questions: list[str] = []

    # Comprehensive queries (200+ chars with multiple sentences) skip clarification.
    has_multiple_points = query.count("(") >= 2 or query.count(",") >= 3
    if query_len >= 200 and has_multiple_points:
        return []

    if query_len < 30:
        questions.append("Can you provide more context about what you're looking for?")

    comparative_terms = ["compare", "vs", "versus", "best", "top"]
    has_comparative = any(term in query_lower for term in comparative_terms)
    if has_comparative and query_len < 100 and not any(c.isdigit() for c in query):
        questions.append("What specific aspects would you like to compare?")
        questions.append("What's your use case or context?")

    has_topic_term = any(term in query_lower for term in ["research", "analyze", "investigate"])
    if has_topic_term and query_len < 100:
        questions.append("What specific angle or focus area interests you?")
        questions.append("What's the timeframe or scope you're interested in?")

    if "best practice" in query_lower and query_len < 100:
        questions.append("What industry or domain are you in?")
        questions.append("What's the scale or context (startup, enterprise, etc.)?")

    return questions[:3]


async def _maybe_clarify_query(
    query: str,
    ctx: Context | None,
) -> str | InputRequiredResult:
    """
    Analyze query and optionally ask clarifying questions.

    Uses heuristics to detect vague queries and prompts for clarification via
    ctx.elicit() (handshake-era foreground calls) or the SEP-2322 guard pattern
    (sessionless protocol / MCP Tasks - see module docstring above).

    Args:
        query: The research query
        ctx: MCP Context (None when running with no context at all)

    Returns the refined query, the original if clarification was skipped/
    unavailable, or an InputRequiredResult if the client must be asked first
    (the tool call ends here; the client re-invokes with the answer).
    """
    if ctx is None:
        logger.info("🔍 Skipping clarification (no context)")
        return query

    questions = _detect_clarifying_questions(query)

    # Comprehensive/specific queries need no clarification at all.
    if not questions:
        logger.info("   ✅ Query is specific enough, no clarification needed")
        return query

    # Resume path: the client already answered this exact guard-pattern round.
    # (ctx.input_responses is only populated on a call that follows an
    # InputRequiredResult; it is None on a fresh call.)
    if ctx.input_responses is not None:
        pending = ctx.input_responses.get(_CLARIFY_INPUT_KEY)
        if pending is not None:
            return _apply_clarification_answer(query, questions, pending)

    logger.info("   🎯 Query may need clarification: %d questions", len(questions))

    if _uses_sessionless_guard_pattern(ctx):
        return _build_clarification_guard(query, questions)

    # Handshake-era foreground path: ctx.elicit() keeps the connection open
    # and blocks for the answer in a single call.
    try:
        from pydantic import create_model

        field_definitions = {
            f"answer_{i + 1}": (str, Field(default="", description=q))
            for i, q in enumerate(questions)
        }
        DynamicSchema = create_model("ClarificationQuestions", **field_definitions)  # type: ignore

        message = (
            f"To improve research quality for:\n\n**\"{query}\"**\n\n"
            f"Please answer these questions (optional - press 'Skip' to continue):"
        )

        result = await ctx.elicit(
            message=message,
            response_type=DynamicSchema,
        )

        if result.action == "accept" and result.data:
            data = result.data.model_dump() if hasattr(result.data, "model_dump") else {}
            answers = [data.get(f"answer_{i + 1}", "") for i in range(len(questions))]
            non_empty = [a for a in answers if a.strip()]

            if non_empty:
                logger.info("   ✨ User provided %d/%d answers", len(non_empty), len(questions))
                clarification = "\n".join(
                    f"Q: {q}\nA: {a}"
                    for q, a in zip(questions, answers, strict=False)
                    if a.strip()
                )
                refined = f"{query}\n\nAdditional context:\n{clarification}"
                logger.info("   📝 Refined query: %s", refined[:100])
                return refined
            else:
                logger.info("   ⏭️ User submitted but answers empty")
        else:
            logger.info("   ⏭️ User skipped/cancelled clarification")

    except Exception as e:
        logger.warning("   ⚠️ Elicitation failed: %s", e)

    return query


# =============================================================================
# Deep Research Tool
# =============================================================================


async def _run_deep_research_tool(
    *,
    query: str,
    format_instructions: str | None = None,
    file_search_store_names: list[str] | None = None,
    mcp_servers: list[dict[str, object]] | None = None,
    visualization: str = "off",
    collaborative_planning: bool = False,
    ctx: Context | None = None,
    agent_name: DeepResearchAgent,
    tool_name: str,
) -> str | InputRequiredResult:
    """
    Comprehensive autonomous research agent (3-20 minutes). Requires MCP
    Tasks support on the client (SEP-1732) — without it, the call is
    rejected with `-32600`. Use for: research reports, competitive
    analysis, "compare X vs Y", "analyze", "investigate", literature
    review, multi-source synthesis.

    For vague queries the tool automatically asks clarifying questions to
    refine the scope before starting (when elicitation is available).

    The session is persisted **at the start** under a stable
    `interaction_id`. If the client disconnects mid-run, call
    `resume_research` with that id to recover; once complete, pipe the
    result into `export_research_session(format="docx")` to save a Word
    deliverable on disk.

    Args:
        query: Research question or topic (can be vague — clarification is
            automatic).
        format_instructions: Optional report structure/tone guidance
        file_search_store_names: Optional file stores for RAG over your own data
        mcp_servers: Disabled compatibility parameter; non-empty values are rejected
        visualization: "off" (default) or "auto" - allow the agent to include
            chart/diagram images in its response
        collaborative_planning: When True, return the drafted research plan (not
            the final report) plus an interaction ID; use refine_research_plan to
            iterate on or approve the plan afterwards

    Returns:
        Comprehensive research report with citations, or (when
        collaborative_planning=True) the drafted plan awaiting approval.
    """
    logger.info("🔬 %s (%s): %s", tool_name, agent_name.value, query[:100])
    try:
        validate_mcp_servers_supported(agent_name=agent_name, mcp_servers=mcp_servers)
    except ValueError as exc:
        raise DeepResearchError(
            "REMOTE_MCP_DISABLED",
            str(exc),
            details={
                "feature": "deep_research_remote_mcp",
                "diagnostic_tool": "inspect_mcp_server_for_gemini",
                "upstream_issue": "https://github.com/googleapis/python-genai/issues/2126",
            },
            category=ErrorCategory.UNSUPPORTED_FEATURE,
        ) from exc

    if format_instructions:
        logger.info("   📝 Format: %s", format_instructions[:80])
    if file_search_store_names:
        logger.info("   📁 File search stores: %s", file_search_store_names)

    # Resolve template key to full template instructions
    effective_format = format_instructions
    if format_instructions:
        from gemini_research_mcp.templates import get_template

        template = get_template(format_instructions)
        if template:
            logger.info("   📋 Using template: %s", template.name)
            effective_format = str(template)

    start = time.time()

    # ==========================================================================
    # Phase 1: Query Clarification (if ctx available)
    # ==========================================================================
    effective_query = await _maybe_clarify_query(query, ctx)

    if isinstance(effective_query, InputRequiredResult):
        # SEP-2322 guard: this call ends here. The client answers and calls
        # this same tool again with the same arguments; ctx.input_responses
        # will then carry the answer and _maybe_clarify_query resumes cleanly.
        logger.info("   ❓ Clarification required (guard pattern) - awaiting client answer")
        return effective_query

    if effective_query != query:
        logger.info("   ✨ Using refined query")
        logger.info("=" * 60)
        logger.info("📋 FINAL CONSOLIDATED QUERY:")
        for line in effective_query.split("\n"):
            logger.info("   %s", line)
        logger.info("=" * 60)

    # ==========================================================================
    # Phase 2: Deep Research Execution
    # ==========================================================================
    if ctx is not None:
        await ctx.report_progress(
            progress=0,
            total=100,
            message=f"Starting {tool_name}...",
        )

    try:
        thought_count = 0
        action_count = 0
        text_parts: list[str] = []
        image_events: list[dict[str, object]] = []
        interaction_id: str | None = None
        session_saved = False  # Track if we saved session at start
        initial_title: str | None = None  # Generated title for the session
        plan_ready = False

        # Consume the stream to get interaction_id and track progress
        async for event in deep_research_stream(
            query=effective_query,
            format_instructions=effective_format,
            file_search_store_names=file_search_store_names,
            mcp_servers=mcp_servers,
            agent_name=agent_name,
            visualization=visualization,
            collaborative_planning=collaborative_planning,
        ):
            if event.interaction_id:
                interaction_id = event.interaction_id
                logger.info("   📋 interaction_id: %s", interaction_id)

                # === RESUME SUPPORT: Save session at START with in_progress status ===
                if not session_saved:
                    try:
                        # Generate a proper title from the query (fast, ~$0.0001)
                        initial_title = await generate_title_from_query(effective_query)
                        if not initial_title:
                            initial_title = effective_query[:60]  # Fallback
                        logger.info("   📝 Generated title: %s", initial_title)

                        save_research_session(
                            interaction_id=interaction_id,
                            query=effective_query,
                            title=initial_title,
                            format_instructions=format_instructions,
                            agent_name=agent_name,
                            status=(
                                ResearchStatus.PLANNING
                                if collaborative_planning
                                else ResearchStatus.IN_PROGRESS
                            ),
                        )
                        session_saved = True
                        logger.info("   💾 Session saved (in_progress) for resume support")
                    except Exception as save_error:
                        logger.warning("⚠️ Failed to save session at start: %s", save_error)

            # Track events for progress
            if event.event_type == "thought":
                thought_count += 1
                content = event.content or ""
                short = content[:55] + "..." if len(content) > 55 else content
                if ctx:
                    await ctx.report_progress(
                        progress=min(50, thought_count * 5),
                        total=100,
                        message=f"[{thought_count}] 🧠 {short}",
                    )
            elif event.event_type == "action":
                action_count += 1
                content = event.content or ""
                short = content[:55] + "..." if len(content) > 55 else content
                if ctx:
                    await ctx.report_progress(
                        progress=min(50, thought_count * 5 + action_count * 2),
                        total=100,
                        message=f"[{action_count}] 🔍 {short}",
                    )
            elif event.event_type == "text":
                if event.content:
                    text_parts.append(event.content)
            elif event.event_type == "image":
                # Never folded into text_parts - collected separately and
                # persisted as export artifacts once the interaction_id is known.
                image_events.append({
                    "data": event.content,
                    "mime_type": event.image_mime_type,
                    "uri": event.image_uri,
                })
            elif event.event_type == "start":
                if ctx:
                    await ctx.report_progress(
                        progress=0,
                        total=100,
                        message="🚀 Research started",
                    )
            elif event.event_type == "plan_ready":
                plan_ready = True
                break
            elif event.event_type == "error":
                error_content = str(event.content or "Deep Research stream error")
                logger.error("   Stream error: %s", error_content)
                retryable = is_retryable_error(error_content)
                # Preserve recoverability for transient stream failures.
                if interaction_id and session_saved:
                    with contextlib.suppress(Exception):
                        update_research_session(
                            interaction_id,
                            status=(
                                ResearchStatus.INTERRUPTED
                                if retryable
                                else ResearchStatus.FAILED
                            ),
                        )
                if retryable and interaction_id:
                    raise DeepResearchError(
                        code="RESEARCH_INTERRUPTED",
                        message=(
                            f"Deep Research stream was interrupted by a retryable error: "
                            f"{error_content}. Interaction ID: {interaction_id}. "
                            f"{_resume_hint(interaction_id)}"
                        ),
                        details={"interaction_id": interaction_id},
                    )
                raise DeepResearchError(
                    code="RESEARCH_FAILED",
                    message=error_content,
                    details={"interaction_id": interaction_id},
                )

        if not interaction_id:
            raise ValueError("No interaction_id received from stream")

        if plan_ready:
            plan_text = "".join(text_parts).strip()
            try:
                if session_saved:
                    update_research_session(
                        interaction_id,
                        status=ResearchStatus.AWAITING_APPROVAL,
                        plan_text=plan_text,
                    )
                else:
                    save_research_session(
                        interaction_id=interaction_id,
                        query=effective_query,
                        title=initial_title or effective_query[:60],
                        format_instructions=format_instructions,
                        agent_name=agent_name,
                        status=ResearchStatus.AWAITING_APPROVAL,
                        plan_text=plan_text,
                    )
                logger.info("   💾 Session updated (awaiting_approval)")
            except Exception as save_error:
                logger.warning(
                    "⚠️ Failed to update session (plan ready): %s", save_error
                )
            return _format_plan_response(plan_text, interaction_id)

        logger.info("   📊 Stream consumed: %d thoughts, %d actions", thought_count, action_count)

        return await _poll_deep_research_to_completion(
            interaction_id=interaction_id,
            effective_query=effective_query,
            start=start,
            text_parts=text_parts,
            image_events=image_events,
            session_saved=session_saved,
            ctx=ctx,
        )

    except DeepResearchError:
        raise
    except Exception as e:
        logger.exception("%s failed: %s", tool_name, e)
        raise DeepResearchError(
            code="INTERNAL_ERROR",
            message=str(e),
        ) from e


async def _poll_deep_research_to_completion(
    *,
    interaction_id: str,
    effective_query: str,
    start: float,
    text_parts: list[str],
    image_events: list[dict[str, object]],
    session_saved: bool,
    ctx: Context | None,
) -> str:
    """Poll a running Deep Research interaction until it completes, fails, or times out.

    Shared by `_run_deep_research_tool` (initial run) and `refine_research_plan`
    (post-approval execution) so both paths persist images, update the stored
    session, and format the final report identically.
    """
    if ctx:
        await ctx.report_progress(
            progress=50,
            total=100,
            message="⏳ Waiting for completion...",
        )

    # Poll for completion
    max_wait = DEEP_RESEARCH_POLL_MAX_WAIT_SECONDS
    poll_interval = DEEP_RESEARCH_POLL_INTERVAL_SECONDS
    poll_start = time.time()

    while time.time() - poll_start < max_wait:
        result = await get_research_status(interaction_id)

        raw_status = "unknown"
        if result.raw_interaction:
            raw_status = getattr(result.raw_interaction, "status", "unknown")

        elapsed = time.time() - start

        if raw_status == "completed":
            logger.info("   ✅ Research completed in %s", _format_duration(elapsed))

            streamed_text = "".join(text_parts)
            if not (result.text or "").strip() and streamed_text.strip():
                result.text = streamed_text

            result = await process_citations(result, resolve_urls=True)

            # Persist any images (agent_config.visualization="auto"). Prefer the
            # final/non-streaming interaction's images (fully-formed content
            # items) and fall back to what was collected while streaming.
            final_images = result.images or image_events
            image_export_ids, image_uris = await _persist_deep_research_images(
                session_id=interaction_id, images=final_images
            )

            # Auto-save session for later follow-up
            total_tokens = None
            if result.usage and result.usage.total_tokens:
                total_tokens = result.usage.total_tokens

            # Generate title and summary in one call (~$0.0003/call)
            metadata = await generate_session_metadata(
                text=result.text or "",
                query=effective_query,
            )

            # Update session with completion data (session saved at start)
            try:
                update_research_session(
                    interaction_id,
                    title=metadata.title or None,
                    summary=metadata.summary or None,
                    report_text=result.text,
                    duration_seconds=elapsed,
                    total_tokens=total_tokens,
                    status=ResearchStatus.COMPLETED,
                    image_export_ids=image_export_ids,
                )
                logger.info("   💾 Session updated (completed)")
            except Exception as save_error:
                logger.warning(
                    "⚠️ Failed to update session (research succeeded): %s",
                    save_error,
                )

            return _format_deep_research_report(result, interaction_id, elapsed, image_uris)

        elif raw_status in ("failed", "cancelled", "canceled"):
            logger.error("   ❌ Research %s after %s", raw_status, _format_duration(elapsed))
            # Mark session with appropriate status
            if session_saved:
                session_status = (
                    ResearchStatus.CANCELLED
                    if raw_status in ("cancelled", "canceled")
                    else ResearchStatus.FAILED
                )
                with contextlib.suppress(Exception):
                    update_research_session(
                        interaction_id,
                        status=session_status,
                        duration_seconds=elapsed,
                    )
            raise DeepResearchError(
                code=f"RESEARCH_{raw_status.upper()}",
                message=f"Research {raw_status} after {_format_duration(elapsed)}",
            )
        else:
            # Still working - report progress
            if ctx:
                progress_pct = min(90, int(50 + (elapsed / max_wait) * 40))
                await ctx.report_progress(
                    progress=progress_pct,
                    total=100,
                    message=f"⏳ Researching... ({_format_duration(elapsed)})",
                )

        await asyncio.sleep(poll_interval)

    # Timeout
    elapsed = time.time() - start
    if session_saved:
        with contextlib.suppress(Exception):
            update_research_session(
                interaction_id,
                status=ResearchStatus.INTERRUPTED,
                duration_seconds=elapsed,
            )
    return "\n".join(
        [
            "## Research Still Running",
            "",
            (
                f"Deep Research did not finish within {_format_duration(elapsed)}, "
                "but the Gemini interaction was saved for recovery."
            ),
            "",
            f"- Interaction ID: `{interaction_id}`",
            f"- Recovery: {_resume_hint(interaction_id)}",
        ]
    )


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
)
async def inspect_mcp_server_for_gemini(
    url: Annotated[str, "HTTPS MCP server endpoint URL to inspect"],
    name: Annotated[str | None, "Optional display name for the MCP server"] = None,
    headers: Annotated[
        dict[str, str] | None,
        "Optional string-to-string headers used when listing server tools",
    ] = None,
    allowed_tools: Annotated[
        list[str] | None,
        "Optional tool names to verify against the server tool list",
    ] = None,
) -> str:
    """
    Inspect a remote MCP server's reachability and tool schemas.

    This standalone diagnostic lists remote tools and flags schema patterns
    relevant to Gemini compatibility. It does not enable remote MCP in Deep
    Research; that provider integration is currently disabled as unreliable.
    """
    result = await _inspect_mcp_server_for_gemini({
        "name": name,
        "url": url,
        "headers": headers,
        "allowed_tools": allowed_tools,
    })
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    task=TaskConfig(mode="required"),
)
async def research_deep(
    query: Annotated[str, "Research question or topic to investigate thoroughly"],
    format_instructions: Annotated[
        str | None,
        "Optional report format (e.g., 'executive briefing', 'comparison table')",
    ] = None,
    file_search_store_names: Annotated[
        list[str] | None,
        "Optional: Gemini File Search store names to search your own data alongside web",
    ] = None,
    mcp_servers: Annotated[
        list[dict[str, object]] | None,
        (
            "Disabled: remote MCP is currently unreliable for Deep Research. "
            "Any non-empty value fails before network or Gemini API access."
        ),
    ] = None,
    visualization: Annotated[
        Literal["off", "auto"],
        "'auto' allows the agent to include chart/diagram images in its response",
    ] = "off",
    collaborative_planning: Annotated[
        bool,
        (
            "When True, return the drafted research plan (not the final report) plus "
            "an interaction ID; call refine_research_plan afterwards to iterate on or "
            "approve the plan"
        ),
    ] = False,
    ctx: Context | None = None,
) -> str | InputRequiredResult:
    """
    Comprehensive autonomous research agent. Takes 3-20 minutes.

    Uses the fast/default April 2026 Deep Research agent. Use this by default for
    interactive questions, exploratory research, competitive analysis, comparisons,
    and latency/cost-sensitive multi-source synthesis.

    For maximum-comprehensiveness research, use research_deep_max instead.

    For vague queries, the tool automatically asks clarifying questions
    to refine the research scope before starting (when elicitation is available).
    """
    return await _run_deep_research_tool(
        query=query,
        format_instructions=format_instructions,
        file_search_store_names=file_search_store_names,
        mcp_servers=mcp_servers,
        visualization=visualization,
        collaborative_planning=collaborative_planning,
        agent_name=get_deep_research_agent(),
        tool_name="research_deep",
        ctx=ctx,
    )


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    task=TaskConfig(mode="required"),
)
async def research_deep_max(
    query: Annotated[str, "Research question or topic to investigate with maximum depth"],
    format_instructions: Annotated[
        str | None,
        "Optional report format (e.g., 'executive briefing', 'comparison table')",
    ] = None,
    file_search_store_names: Annotated[
        list[str] | None,
        "Optional: Gemini File Search store names to search your own data alongside web",
    ] = None,
    mcp_servers: Annotated[
        list[dict[str, object]] | None,
        (
            "Disabled: remote MCP is currently unreliable for Deep Research Max. "
            "Any non-empty value fails before network or Gemini API access."
        ),
    ] = None,
    visualization: Annotated[
        Literal["off", "auto"],
        "'auto' allows the agent to include chart/diagram images in its response",
    ] = "off",
    collaborative_planning: Annotated[
        bool,
        (
            "When True, return the drafted research plan (not the final report) plus "
            "an interaction ID; call refine_research_plan afterwards to iterate on or "
            "approve the plan"
        ),
    ] = False,
    ctx: Context | None = None,
) -> str | InputRequiredResult:
    """
    Maximum-comprehensiveness autonomous research agent.

    Use when the user explicitly says "Max", "deep research max", "exhaustive",
    "comprehensive", "due diligence", "market map", "literature review",
    "high stakes", "board-ready", "offline/nightly", or asks for maximum
    completeness over speed.

    For ordinary interactive research, use research_deep instead.
    """
    return await _run_deep_research_tool(
        query=query,
        format_instructions=format_instructions,
        file_search_store_names=file_search_store_names,
        mcp_servers=mcp_servers,
        visualization=visualization,
        collaborative_planning=collaborative_planning,
        agent_name=DeepResearchAgent.DEEP_RESEARCH_MAX,
        tool_name="research_deep_max",
        ctx=ctx,
    )


# =============================================================================
# refine_research_plan: Iterate on or approve a collaborative-planning session
# =============================================================================


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    task=TaskConfig(mode="required"),
)
async def refine_research_plan(
    previous_interaction_id: Annotated[
        str,
        (
            "Interaction ID returned by research_deep/research_deep_max when "
            "collaborative_planning=True"
        ),
    ],
    decision: Annotated[
        Literal["iterate", "approve"],
        (
            "'iterate' requests a revised plan (use instructions for feedback); "
            "'approve' executes the approved plan and returns the final report"
        ),
    ],
    instructions: Annotated[
        str | None,
        "Feedback for a new plan iteration, or extra guidance while executing the plan",
    ] = None,
    ctx: Context | None = None,
) -> str:
    """
    Continue a collaborative-planning Deep Research session.

    Call this with the interaction_id returned by research_deep/research_deep_max
    after they returned a plan awaiting approval (collaborative_planning=True).

    - decision="iterate": request a revised plan; returns the new plan text plus
      a new interaction_id to pass back into this tool.
    - decision="approve": execute the approved plan; returns the final report
      once Deep Research completes (this call can take several minutes).
    """
    logger.info(
        "🗂️ refine_research_plan: id=%s decision=%s", previous_interaction_id, decision
    )

    session = get_research_session(previous_interaction_id)
    if session is None:
        return (
            f"❌ No research session found for interaction_id `{previous_interaction_id}`. "
            "Call research_deep(..., collaborative_planning=True) first."
        )
    if session.status != ResearchStatus.AWAITING_APPROVAL:
        return (
            f"❌ Session `{previous_interaction_id}` is not awaiting plan approval "
            f"(current status: `{session.status.value}`)."
        )

    agent_name = session.agent_name or get_deep_research_agent()
    start = time.time()
    text_parts: list[str] = []
    image_events: list[dict[str, object]] = []
    new_interaction_id: str | None = None
    plan_ready = False

    follow_up_query = instructions or (
        "Please revise the plan based on this feedback and share an updated plan."
        if decision == "iterate"
        else "The plan is approved. Please proceed to execute the research and produce "
        "the final report."
    )

    try:
        async for event in deep_research_stream(
            query=follow_up_query,
            agent_name=agent_name,
            visualization="off",
            collaborative_planning=(decision == "iterate"),
            previous_interaction_id=previous_interaction_id,
        ):
            if event.interaction_id:
                new_interaction_id = event.interaction_id

            if event.event_type == "text":
                if event.content:
                    text_parts.append(event.content)
            elif event.event_type == "image":
                image_events.append({
                    "data": event.content,
                    "mime_type": event.image_mime_type,
                    "uri": event.image_uri,
                })
            elif event.event_type == "plan_ready":
                plan_ready = True
                break
            elif event.event_type == "error":
                error_content = str(event.content or "Deep Research stream error")
                logger.error("   Stream error: %s", error_content)
                raise DeepResearchError(
                    code="RESEARCH_STREAM_ERROR", message=error_content
                )

        if not new_interaction_id:
            raise DeepResearchError(
                code="INTERNAL_ERROR",
                message="No interaction_id returned by plan continuation",
            )

        if decision == "iterate" and plan_ready:
            plan_text = "".join(text_parts)
            try:
                save_research_session(
                    interaction_id=new_interaction_id,
                    query=session.query,
                    title=session.title or session.query[:60],
                    format_instructions=session.format_instructions,
                    agent_name=agent_name,
                    status=ResearchStatus.AWAITING_APPROVAL,
                    plan_text=plan_text,
                )
            except Exception as save_error:
                logger.warning(
                    "⚠️ Failed to save refined plan session: %s", save_error
                )
            return _format_plan_response(plan_text, new_interaction_id)

        # decision == "approve" (or the agent proceeded straight to execution).
        try:
            save_research_session(
                interaction_id=new_interaction_id,
                query=session.query,
                title=session.title or session.query[:60],
                format_instructions=session.format_instructions,
                agent_name=agent_name,
                status=ResearchStatus.EXECUTING,
                plan_text=session.plan_text,
            )
        except Exception as save_error:
            logger.warning("⚠️ Failed to save executing session: %s", save_error)

        return await _poll_deep_research_to_completion(
            interaction_id=new_interaction_id,
            effective_query=session.query,
            start=start,
            text_parts=text_parts,
            image_events=image_events,
            session_saved=True,
            ctx=ctx,
        )
    except DeepResearchError:
        raise
    except Exception as e:
        logger.exception("refine_research_plan failed: %s", e)
        raise DeepResearchError(code="INTERNAL_ERROR", message=str(e)) from e


# =============================================================================
# list_format_templates: Show available format instruction templates
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def list_format_templates(
    category: Annotated[
        str | None,
        "Filter by category: 'business', 'analysis', 'technical', or 'academic'",
    ] = None,
) -> str:
    """
    List available format instruction templates for research_deep.

    Pre-built templates help generate consistently structured reports:
    - executive_briefing: C-suite summary with key findings and recommendations
    - competitive_analysis: Deep dive comparison of competitors
    - comparison_table: Side-by-side feature comparison with verdict
    - technical_overview: Technical explanation for engineers
    - literature_review: Academic-style synthesis of research

    Use the template name with research_deep's format_instructions parameter:
    Example: research_deep(query="...", format_instructions="executive_briefing")

    Returns:
        JSON list of available templates with descriptions
    """
    from gemini_research_mcp.templates import ALL_TEMPLATES, TEMPLATES_BY_CATEGORY, TemplateCategory

    logger.info("📋 list_format_templates: category=%s", category)

    if category:
        try:
            cat = TemplateCategory(category.lower())
            templates = TEMPLATES_BY_CATEGORY.get(cat, [])
            template_list = [
                {
                    "key": key,
                    "name": t.name,
                    "description": t.description,
                    "category": t.category.value,
                }
                for key, t in ALL_TEMPLATES.items()
                if t in templates
            ]
        except ValueError:
            return json.dumps({
                "error": f"Invalid category: {category}",
                "valid_categories": [c.value for c in TemplateCategory],
            })
    else:
        template_list = [
            {
                "key": key,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
            }
            for key, t in ALL_TEMPLATES.items()
        ]

    return json.dumps({
        "templates": template_list,
        "count": len(template_list),
        "usage": "Pass template key to research_deep's format_instructions parameter",
        "example": 'research_deep(query="...", format_instructions="executive_briefing")',
    }, indent=2)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def research_followup(
    query: Annotated[
        str, "Follow-up question about previous research (e.g., 'elaborate on surface codes')"
    ],
    interaction_id: Annotated[
        str | None,
        "Optional: specific interaction_id. If not provided, auto-matches from sessions.",
    ] = None,
    model: Annotated[
        str | None, "Model to use for follow-up. Defaults to the configured GEMINI_MODEL."
    ] = None,
) -> str:
    """
    Continue conversation after deep research. Ask follow-up questions without restarting.

    The tool automatically finds the relevant research session based on your question.
    You can optionally provide an interaction_id for direct reference.

    Use for: "clarify", "elaborate", "summarize", "explain more", "what about",
    continue discussion, ask more questions about completed research results.

    Args:
        query: Your follow-up question
        interaction_id: Optional specific session ID (from list_research_sessions)
        model: Model to use (default: configured GEMINI_MODEL / gemini-3.8-flash)

    Returns:
        Response to the follow-up question
    """
    logger.info("💬 research_followup: query=%s, id=%s", query[:100], interaction_id)

    try:
        # If no interaction_id provided, find the best matching session
        previous_interaction_id = interaction_id
        if not previous_interaction_id:
            sessions = _list_sessions(limit=20, include_expired=False)
            if not sessions:
                return "❌ No research sessions found. Complete a deep research first."

            # Filter out cancelled and failed sessions for auto-matching
            matchable = [
                s for s in sessions
                if s.status not in (ResearchStatus.CANCELLED, ResearchStatus.FAILED)
            ]
            if not matchable:
                return "❌ No active research sessions found. All sessions are cancelled or failed."

            # Build session list for semantic matching
            session_dicts = [
                {
                    "id": s.interaction_id,
                    "query": s.query,
                    "summary": s.summary or s.query[:100],
                }
                for s in matchable
            ]

            matched_id = await semantic_match_session(query, session_dicts)
            if matched_id:
                previous_interaction_id = matched_id
                # Find the matched session for logging
                matched_session = next(
                    (s for s in sessions if s.interaction_id == matched_id), None
                )
                if matched_session:
                    logger.info(
                        "   📎 Matched to session: %s (%s)",
                        matched_id[:12],
                        matched_session.query[:50],
                    )
            else:
                # Fall back to most recent matchable session
                previous_interaction_id = matchable[0].interaction_id
                logger.info(
                    "   📎 No semantic match, using most recent: %s (%s)",
                    matchable[0].interaction_id[:12],
                    matchable[0].query[:50],
                )

        response = await _research_followup(
            previous_interaction_id=previous_interaction_id,
            query=query,
            model=model,
        )

        lines = [
            "## Follow-up Response",
            "",
            response,
            "",
            "---",
            f"*Interaction ID: `{previous_interaction_id}`*",
        ]

        return "\n".join(lines)

    except Exception as e:
        logger.exception("research_followup failed: %s", e)
        return f"❌ Follow-up failed: {e}"


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def list_research_sessions(
    limit: Annotated[int, "Maximum number of sessions to return"] = 20,
    include_expired: Annotated[bool, "Include expired sessions"] = False,
) -> str:
    """
    List saved research sessions available for follow-up.

    Sessions are automatically saved when deep research completes successfully.
    Returns JSON for easy parsing by agents.

    Note: You don't need to extract interaction_ids manually.
    Just use research_followup with your question - it will automatically
    find the matching session.

    Returns:
        JSON array of research sessions with summaries
    """
    logger.info("📋 list_research_sessions: limit=%d, include_expired=%s", limit, include_expired)

    sessions = _list_sessions(limit=limit, include_expired=include_expired)

    if not sessions:
        return json.dumps({"sessions": [], "message": "No research sessions found."})

    session_list = []
    for session in sessions:
        session_data: dict[str, str | int | float | None] = {
            "interaction_id": session.interaction_id,
            "query": session.query,
            "summary": session.summary,
            "status": session.status.value,
            "created_at": session.created_at_iso,
            "expires_in": session.time_remaining_human,
        }
        if session.title:
            session_data["title"] = session.title
        if session.duration_seconds:
            session_data["duration_seconds"] = session.duration_seconds
        if session.total_tokens:
            session_data["total_tokens"] = session.total_tokens

        session_list.append(session_data)

    # Count resumable sessions
    resumable_count = sum(1 for s in sessions if s.is_resumable)
    hint = "Use research_followup with your question - auto-matches session."
    if resumable_count > 0:
        hint = f"{resumable_count} session(s) can be resumed with resume_research tool."

    return json.dumps(
        {
            "sessions": session_list,
            "count": len(session_list),
            "resumable_count": resumable_count,
            "hint": hint,
        },
        indent=2,
    )


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def resume_research(
    interaction_id: Annotated[
        str | None,
        "Optional: specific interaction_id to resume. If not provided, shows resumable sessions.",
    ] = None,
    ctx: Context | None = None,
) -> str:
    """
    Resume an interrupted or in-progress `research_deep` session.

    Use when a `research_deep` call was cut short (client disconnect,
    transport error, cancellation) or when you want to check whether a
    long-running session has since completed on Gemini's servers.

    Because `research_deep` persists its session at the start, the
    research continues on Gemini's side even when the MCP client goes
    away — this tool retrieves the result once it's ready.

    Call with no arguments to list recoverable sessions. Call with an
    `interaction_id` (returned by the original `research_deep` call or by
    `list_research_sessions`) to check a specific session's status; if it
    has completed, the full report is returned. Hand the result off to
    `export_research_session(format="docx")` to save the report to disk.

    Args:
        interaction_id: Optional session to resume/check. Omit to list
            recoverable sessions.

    Returns:
        JSON describing recoverable sessions, or the completed research
        report for the given `interaction_id`.
    """
    logger.info("🔄 resume_research: id=%s", interaction_id)

    try:
        # If no interaction_id, list all resumable sessions
        if not interaction_id:
            sessions = _list_sessions(limit=50, include_expired=False)
            now = time.time()
            resumable = [s for s in sessions if s.is_resumable]
            recent_resumable = [
                s for s in resumable if now - s.created_at <= STALE_RESUMABLE_SECONDS
            ][:10]
            stale_resumable = [
                s for s in resumable if now - s.created_at > STALE_RESUMABLE_SECONDS
            ][:10]
            recent_failed = [
                s
                for s in sessions
                if s.status == ResearchStatus.FAILED
                and now - s.created_at <= RECENT_FAILED_SECONDS
            ][:10]

            if not recent_resumable and not stale_resumable and not recent_failed:
                return json.dumps({
                    "status": "no_resumable_sessions",
                    "message": "No interrupted or in-progress research sessions found.",
                    "hint": (
                        "All sessions are completed/expired, or the failed run used a "
                        "different GEMINI_RESEARCH_STORAGE_PATH or MCP runtime."
                    ),
                })

            def summarize_session(session: ResearchSession) -> dict[str, str | float | bool]:
                age_hours = (now - session.created_at) / 3600
                return {
                    "interaction_id": session.interaction_id,
                    "query": session.query[:100],
                    "status": session.status.value,
                    "created_at": session.created_at_iso,
                    "expires_in": session.time_remaining_human or "unknown",
                    "age_hours": round(age_hours, 1),
                    "stale": age_hours > 24,
                }

            return json.dumps({
                "status": (
                    "resumable_sessions_found"
                    if recent_resumable
                    else (
                        "recent_failed_sessions_found"
                        if recent_failed
                        else "stale_resumable_sessions_found"
                    )
                ),
                "count": len(recent_resumable),
                "sessions": [summarize_session(s) for s in recent_resumable],
                "stale_count": len(stale_resumable),
                "stale_sessions": [summarize_session(s) for s in stale_resumable],
                "recent_failed_count": len(recent_failed),
                "recent_failed_sessions": [
                    summarize_session(s) for s in recent_failed
                ],
                "hint": (
                    "Call resume_research with a specific interaction_id. "
                    "Recent failed sessions are included because retryable gateway "
                    "timeouts can be misclassified by older server versions."
                ),
            }, indent=2)

        # Check specific session status
        session = get_research_session(interaction_id)
        if not session:
            return json.dumps({
                "status": "not_found",
                "message": f"Session not found: {interaction_id}",
            })

        # If already completed, return the report
        if session.status == ResearchStatus.COMPLETED:
            if (session.report_text or "").strip():
                return json.dumps({
                    "status": "already_completed",
                    "message": "This research session is already completed.",
                    "title": session.title,
                    "summary": session.summary,
                    "hint": "Use research_followup or export_research_session.",
                })
            logger.info(
                "Completed session %s has no stored report; checking Gemini status",
                interaction_id[:12],
            )

        # If already cancelled, don't re-query — it's terminal
        if session.status == ResearchStatus.CANCELLED:
            return json.dumps({
                "status": "cancelled",
                "message": "This research was cancelled and cannot be resumed.",
                "resumable": False,
                "query": session.query,
                "hint": "Start a new research_deep with your query.",
            })

        # Check with Gemini API for current status
        if ctx:
            await ctx.info(f"Checking status of research: {session.query[:50]}...")

        try:
            result = await get_research_status(interaction_id)
            raw_status = "unknown"
            if result.raw_interaction:
                raw_status = getattr(result.raw_interaction, "status", "unknown")

            if raw_status == "completed":
                # Research completed on Gemini's side - update our records
                result = await process_citations(result, resolve_urls=True)

                total_tokens = None
                if result.usage and result.usage.total_tokens:
                    total_tokens = result.usage.total_tokens

                if not (result.text or "").strip():
                    update_research_session(
                        interaction_id,
                        total_tokens=total_tokens,
                        status=ResearchStatus.INTERRUPTED,
                    )
                    return json.dumps({
                        "status": "completed_without_report",
                        "message": (
                            "Gemini reports this interaction completed, but no report text "
                            "was returned. The session remains marked interrupted so it "
                            "stays visible for recovery checks."
                        ),
                        "resumable": True,
                        "query": session.query[:100],
                        "interaction_id": interaction_id,
                        "hint": (
                            "Try resume_research again later. If the report remains empty, "
                            "start a new research_deep_max run."
                        ),
                    }, indent=2)

                # Generate metadata
                metadata = await generate_session_metadata(
                    text=result.text or "",
                    query=session.query,
                )

                # Update session
                update_research_session(
                    interaction_id,
                    title=metadata.title or None,
                    summary=metadata.summary or None,
                    report_text=result.text,
                    total_tokens=total_tokens,
                    status=ResearchStatus.COMPLETED,
                )

                logger.info("   ✅ Research recovered and saved!")

                # Return the report
                lines = ["## Research Report (Resumed)"]
                if result.text:
                    lines.append(result.text)
                else:
                    lines.append("*No report text available.*")

                lines.extend([
                    "",
                    "---",
                    f"*Session recovered. Interaction ID: `{interaction_id}`*",
                ])

                return "\n".join(lines)

            elif raw_status in ("failed", "cancelled", "canceled"):
                session_status = (
                    ResearchStatus.CANCELLED
                    if raw_status in ("cancelled", "canceled")
                    else ResearchStatus.FAILED
                )
                update_research_session(
                    interaction_id,
                    status=session_status,
                )
                return json.dumps({
                    "status": raw_status,
                    "message": f"Research {raw_status} on Gemini's servers.",
                    "resumable": False,
                    "query": session.query,
                })

            else:
                # Still in progress — check if stale (>24h)
                age_hours = (time.time() - session.created_at) / 3600
                if age_hours > 24:
                    delete_research_session(interaction_id)
                    logger.info(
                        "🗑️ Deleted stale session %s (%.0fh old)",
                        interaction_id[:12],
                        age_hours,
                    )
                    return json.dumps({
                        "status": "deleted_stale",
                        "message": f"Session deleted — stuck in progress for {age_hours:.0f}h.",
                        "query": session.query[:100],
                    })

                return json.dumps({
                    "status": "still_in_progress",
                    "gemini_status": raw_status,
                    "message": "Research is still running on Gemini's servers.",
                    "query": session.query[:100],
                    "hint": "Try again in a few minutes.",
                })

        except Exception as api_error:
            error_str = str(api_error).lower()
            # If Gemini says not found / gone, delete local session
            if "not_found" in error_str or "404" in error_str:
                delete_research_session(interaction_id)
                logger.info(
                    "🗑️ Deleted session %s — no longer exists on Gemini",
                    interaction_id[:12],
                )
                return json.dumps({
                    "status": "deleted_not_found",
                    "message": "Session no longer exists on Gemini — deleted locally.",
                    "query": session.query[:100],
                })

            logger.warning("Failed to check Gemini status: %s", api_error)
            # Mark as interrupted if we can't reach Gemini
            update_research_session(
                interaction_id,
                status=ResearchStatus.INTERRUPTED,
            )
            return json.dumps({
                "status": "api_error",
                "message": f"Could not check status: {api_error}",
                "hint": "The research may still be running. Try again later.",
            })

    except Exception as e:
        logger.exception("resume_research failed: %s", e)
        return json.dumps({"error": f"Resume failed: {e}"})


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def export_research_session(
    interaction_id: Annotated[
        str | None,
        "Interaction ID of the session to export. If not provided, exports the most recent.",
    ] = None,
    format: Annotated[
        str,
        "Export format: 'markdown' (.md), 'json' (.json), or 'docx' (Word document)",
    ] = "markdown",
    query: Annotated[
        str | None,
        "Optional: search for a session by query text instead of interaction_id",
    ] = None,
    output_path: Annotated[
        str | None,
        (
            "Filesystem path to save the exported file to. Absolute or relative "
            "to the MCP server's working directory. If omitted, the file is "
            "automatically written to GEMINI_RESEARCH_EXPORT_DIR "
            "(default ~/.gemini-research/exports/) and the resolved path is "
            "returned in the response. Parent directory must already exist "
            "when an explicit path is supplied."
        ),
    ] = None,
) -> str | list[TextContent | EmbeddedResource]:
    """
    Save (export, download, archive) a completed research session to a file
    on disk — Word (.docx), Markdown (.md), or JSON. Use this to recover
    your report after a `research_deep` run completes or is interrupted,
    and to convert reports into a shareable Word document.

    **Disk-first contract.** The file is always written to disk, and the
    absolute path is returned on the first line of the response as
    `Saved to: <path>`. When `output_path` is omitted the file lands in
    `GEMINI_RESEARCH_EXPORT_DIR` (default ``~/.gemini-research/exports/``).
    An `EmbeddedResource` is still attached so GUI hosts can expose their
    native "Save As" affordance.

    Similar to Google's Deep Research export feature, the DOCX output is a
    professional Word document suitable for sharing, archiving, or further
    editing.

    Supported formats:
    - **docx**: Word document with headings, lists, and table of contents
    - **markdown**: Clean `.md` file with full report and citations
    - **json**: Machine-readable, all metadata preserved

    Typical recovery flow:
    1. `research_deep(...)` completes (or is resumed via `resume_research`).
    2. `export_research_session(format="docx")` — no other arguments
       needed; the path on disk is returned in the response text.

    Args:
        interaction_id: Specific session ID (from `list_research_sessions`
            or from the `research_deep` / `resume_research` response).
        format: `"docx"`, `"markdown"`, or `"json"`.
        query: Search for a session by query text (alternative to
            `interaction_id`).
        output_path: Optional explicit filesystem path. Parent directory
            must exist. When omitted, a path under the export directory is
            chosen automatically.

    Returns:
        A text block containing ``Saved to: <absolute path>`` plus an
        `EmbeddedResource` carrying the file bytes.
    """
    logger.info(
        "📤 export_research_session: id=%s, format=%s, query=%s, output_path=%s",
        interaction_id,
        format,
        query[:30] if query else None,
        output_path,
    )

    try:
        session = None

        # Find session by interaction_id
        if interaction_id:
            session = get_research_session(interaction_id)
            if not session:
                return json.dumps({
                    "error": f"Session not found: {interaction_id}",
                    "hint": "Use list_research_sessions to see available sessions.",
                })

        # Find session by query using AI-powered semantic matching
        elif query:
            sessions = _list_sessions(limit=20, include_expired=False)
            if not sessions:
                return json.dumps({
                    "error": "No research sessions found.",
                    "hint": "Complete a deep research first with research_deep.",
                })

            # Build session list for semantic matching (same as research_followup)
            session_dicts = [
                {
                    "id": s.interaction_id,
                    "query": s.query,
                    "summary": s.summary or s.query[:100],
                }
                for s in sessions
            ]

            matched_id = await semantic_match_session(query, session_dicts)
            if matched_id:
                session = next((s for s in sessions if s.interaction_id == matched_id), None)
                if session:
                    logger.info(
                        "   📎 Matched to session: %s (%s)",
                        matched_id[:12],
                        session.query[:50],
                    )
                else:
                    # Matched ID not found in sessions list - fall back to most recent
                    session = sessions[0]
                    logger.warning(
                        "   ⚠️ Matched ID %s not in sessions, using most recent",
                        matched_id[:12],
                    )
            else:
                # Fall back to most recent session
                session = sessions[0]
                logger.info(
                    "   📎 No semantic match, using most recent: %s (%s)",
                    session.interaction_id[:12],
                    session.query[:50],
                )

        # Default to most recent session
        else:
            sessions = _list_sessions(limit=1)
            if not sessions:
                return json.dumps({
                    "error": "No research sessions found.",
                    "hint": "Complete a deep research first with research_deep.",
                })
            session = sessions[0]

        # Export — default to disk-first when no output_path is supplied so
        # clients that cannot render EmbeddedResource blobs (headless/CLI)
        # still receive the file at a known location. GUI clients keep the
        # EmbeddedResource for their native "Save As" affordance.
        resolved_path: Path | None = None
        auto_defaulted = False
        if output_path is not None:
            resolved_path = Path(output_path).expanduser().resolve()
            if not resolved_path.parent.exists():
                return json.dumps({
                    "error": (
                        f"Parent directory does not exist: {resolved_path.parent}"
                    ),
                    "hint": (
                        "Create the directory first (e.g. `mkdir -p`) or pass a "
                        "path whose parent already exists."
                    ),
                })
        else:
            # Compute deterministic default path inside the export directory.
            # We need the result filename first — run the export without a
            # path, then persist to disk ourselves and return the path.
            auto_defaulted = True

        if auto_defaulted:
            result = export_session(session, format)
            try:
                export_dir = get_export_dir()
                resolved_path = (export_dir / result.filename).resolve()
                resolved_path.write_bytes(result.content)
                logger.info(
                    "📄 Auto-exported to %s (%s)", resolved_path, result.size_human
                )
            except OSError as write_err:
                logger.warning(
                    "Failed to write default export path: %s", write_err
                )
                resolved_path = None
        else:
            result = export_session(session, format, output_path=resolved_path)

        # Persist every export so research://exports/{id} survives process restarts
        # and is shared by workers when the Redis backend is configured.
        export_id = await _cache_export(result, session.interaction_id)

        # Return EmbeddedResource for all formats to enable VS Code "Save As" button
        # Following the ElevenLabs MCP pattern: put filename in URI for browser-like save dialog
        import base64

        # URI contains filename so clients extract it for "Save As" dialog (like browsers)
        resource_uri = f"research://exports/{export_id}"

        # For text formats (MD, JSON), use TextResourceContents
        # For binary formats (DOCX), use BlobResourceContents
        if result.format == ExportFormat.DOCX:
            resource_content: BlobResourceContents | TextResourceContents = BlobResourceContents(
                uri=resource_uri,
                mime_type=result.mime_type,
                blob=base64.b64encode(result.content).decode("ascii"),
            )
        else:
            # Text formats - use TextResourceContents
            resource_content = TextResourceContents(
                uri=resource_uri,
                mime_type=result.mime_type,
                text=result.content.decode("utf-8"),
            )

        embedded = EmbeddedResource(
            type="resource",
            resource=resource_content,
        )

        # Format-specific emoji and label
        format_info = {
            ExportFormat.DOCX: ("📄", "DOCX"),
            ExportFormat.MARKDOWN: ("📝", "Markdown"),
            ExportFormat.JSON: ("📋", "JSON"),
        }
        emoji, label = format_info.get(result.format, ("📁", result.format.value.upper()))

        # Return metadata as TextContent + file as EmbeddedResource.
        # The "Saved to:" line is emitted first so headless/CLI clients that
        # cannot render the EmbeddedResource blob can still locate the file
        # on disk. GUI clients (e.g. VS Code) use the embedded resource for
        # their native "Save As" affordance.
        if resolved_path is not None:
            saved_header = f"✅ **Saved to:** `{resolved_path}`\n\n"
        else:
            saved_header = (
                "⚠️ **Not saved to disk** — file delivered via embedded "
                "resource only. Re-run with an explicit `output_path` to "
                "materialize the file.\n\n"
            )
        metadata_text = (
            f"{saved_header}"
            f"{emoji} **{label} Export Complete**\n\n"
            f"- **Filename:** {result.filename}\n"
            f"- **Size:** {result.size_human}\n"
            f"- **Session:** {session.query[:80]}\n"
            f"- **Resource URI:** {resource_uri}\n\n"
            f"The file is also attached below as an embedded resource."
        )
        text_content = TextContent(
            type="text",
            text=metadata_text,
        )

        return [text_content, embedded]

    except ImportError as e:
        return json.dumps({
            "error": str(e),
            "hint": "Install skelmis-docx for DOCX export.",
        })
    except Exception as e:
        logger.exception("export_research_session failed: %s", e)
        return json.dumps({"error": f"Export failed: {e}"})


# =============================================================================
# Resources
# =============================================================================


@mcp.resource("research://models")
def get_research_models() -> str:
    """
    List available research models and their capabilities.

    Returns information about the models used by this server:
    - Quick research model (Gemini + Google Search grounding)
    - Deep Research Agent (autonomous multi-step research)
    """
    quick_model = get_model()
    deep_agent = get_deep_research_agent()

    return f"""# Available Research Models

## Quick Research (research_web)

**Model:** `{quick_model}`
- **Latency:** 5-30 seconds
- **API:** Gemini + Google Search grounding
- **Best for:** Fact-checking, current events, quick lookups, documentation
- **Features:** Real-time web search, thinking summaries

## Deep Research (research_deep)

**Agent:** `{deep_agent}`
- **Latency:** 3-20 minutes (can take up to 60 min for complex topics)
- **API:** Gemini Interactions API (Deep Research Agent)
- **Best for:** Research reports, competitive analysis, literature reviews
- **Features:**
  - Autonomous multi-step investigation
  - Built-in Google Search and URL analysis
  - Cited reports with sources
  - File search (RAG) with `file_search_store_names`
  - Format instructions for custom output structure

## Follow-up (research_followup)

**Model:** Configurable (default: gemini-3.8-flash)
- **Latency:** 5-30 seconds
- **API:** Gemini Interactions API
- **Best for:** Clarification, elaboration, summarization of prior research
- **Requires:** `previous_interaction_id` from completed research
"""


@mcp.resource(
    "research://exports/{export_id}",
    name="Research Export",
    description="Download an exported research report. Use export_research_session tool first.",
)
async def get_export_by_id(export_id: str) -> BlobResourceContents | TextResourceContents:
    """
    Retrieve an exported research report by its export ID.

    The export_research_session tool creates exports and returns resource URIs.
    This resource serves the content for download with proper MIME type.

    In VS Code Copilot, you can:
    - Click "Save" to download the file
    - Drag-and-drop from chat to your workspace

    Args:
        export_id: The unique export identifier from export_research_session

    Returns:
        Resource content with proper MIME type (Markdown, JSON, or DOCX)
    """
    import base64

    entry = await _get_cached_export(export_id)
    if not entry:
        raise ValueError(f"Export not found or expired: {export_id}")

    logger.info("📥 Serving export %s (%s)", export_id, entry.filename)

    uri = f"research://exports/{export_id}"
    mime_type = entry.mime_type

    # For text formats, return TextResourceContents
    if mime_type in ("text/markdown", "application/json"):
        return TextResourceContents(
            uri=uri,
            mime_type=mime_type,
            text=entry.content.decode("utf-8"),
        )

    # For binary formats (DOCX), return BlobResourceContents with base64
    return BlobResourceContents(
        uri=uri,
        mime_type=mime_type,
        blob=base64.b64encode(entry.content).decode("ascii"),
    )


@mcp.resource(
    "research://exports",
    name="Available Exports",
    description="List all currently cached exports ready for download.",
    mime_type="application/json",
)
async def list_exports() -> str:
    """
    List all currently available exports.

    Returns a JSON array of available exports with their metadata.
    Exports expire after 1 hour (backend-enforced TTL).
    """
    artifacts: list[ExportArtifact] = await get_export_store().list_async()

    exports = []
    for entry in artifacts:
        remaining = (entry.created_at + EXPORT_TTL_SECONDS) - time.time()
        exports.append({
            "export_id": entry.export_id,
            "uri": f"research://exports/{entry.export_id}",
            "filename": entry.filename,
            "format": entry.format,
            "size": entry.size_human,
            "mime_type": entry.mime_type,
            "session_id": entry.session_id[:12] + "...",
            "expires_in": f"{max(0, int(remaining))}s",
        })

    return json.dumps({"exports": exports, "count": len(exports)}, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================


# =============================================================================
# Transport Configuration (dual transport: stdio default, opt-in streamable-http)
# =============================================================================

# Hosts considered local-only. 0.0.0.0/:: bind all interfaces and are NOT
# loopback - they require explicit authentication before the server will start.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback_host(host: str) -> bool:
    """True if `host` only accepts connections from the local machine."""
    return host in _LOOPBACK_HOSTS


def _require_auth_for_non_loopback(*, host: str, has_auth: bool) -> None:
    """Refuse to start streamable-http on a non-loopback host without auth.

    A remote, unauthenticated HTTP endpoint would let anyone consume the
    server owner's Gemini API quota. Fail closed rather than silently
    exposing the server.
    """
    if not _is_loopback_host(host) and not has_auth:
        raise SystemExit(
            f"Refusing to bind streamable-http to non-loopback host {host!r} without "
            "authentication configured. Set GEMINI_RESEARCH_HTTP_BEARER_TOKEN (or "
            "assign a custom fastmcp AuthProvider to `mcp.auth`) before exposing this "
            "server beyond localhost, or bind to 127.0.0.1/localhost for local-only access."
        )


def main() -> None:
    """Run the MCP server on stdio (default) or opt-in streamable-http transport."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="gemini-research-mcp",
        description="Gemini Research MCP Server - AI-powered research tools",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.environ.get("GEMINI_RESEARCH_TRANSPORT", "stdio"),
        help=(
            "Transport protocol (default: stdio, the historical VS Code/Claude "
            "Desktop config). streamable-http is opt-in, for remote/multi-worker "
            "deployments (env: GEMINI_RESEARCH_TRANSPORT)."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GEMINI_RESEARCH_HTTP_HOST", "127.0.0.1"),
        help=(
            "Host to bind for streamable-http transport. Defaults to 127.0.0.1 "
            "(loopback-only). Binding elsewhere requires authentication "
            "(env: GEMINI_RESEARCH_HTTP_HOST)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GEMINI_RESEARCH_HTTP_PORT", "8000")),
        help="Port for streamable-http transport (default: 8000, env: GEMINI_RESEARCH_HTTP_PORT)",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("GEMINI_RESEARCH_HTTP_PATH", "/mcp"),
        help=(
            "URL path for streamable-http transport (default: /mcp, "
            "env: GEMINI_RESEARCH_HTTP_PATH)"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Set API key from CLI flag if provided (overrides env var)
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key

    logger.info("🚀 Starting Gemini Research MCP Server v%s (MCP SDK)", __version__)
    logger.info("   Task mode: enabled (MCP Tasks / SEP-1732)")

    if args.transport == "stdio":
        logger.info("   Transport: stdio")
        mcp.run(transport="stdio")
        return

    # GEMINI_API_KEY must never double as an MCP client credential - it is a
    # provider secret, not an access-control token for this server's clients.
    bearer_token = os.environ.get("GEMINI_RESEARCH_HTTP_BEARER_TOKEN")
    if bearer_token:
        from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

        # FastMCP's documented static-token verifier: a simple shared-secret
        # bearer check suitable for a single trusted client/deployment. For
        # multi-user or production-grade auth, assign a full AuthProvider
        # (OAuthProvider/JWTVerifier/etc.) to `mcp.auth` instead.
        mcp.auth = StaticTokenVerifier(
            tokens={bearer_token: {"client_id": "gemini-research-mcp-client", "scopes": []}}
        )

    _require_auth_for_non_loopback(host=args.host, has_auth=mcp.auth is not None)

    logger.info("   Transport: streamable-http (http://%s:%s%s)", args.host, args.port, args.path)
    logger.info("   Auth: %s", "bearer token required" if mcp.auth else "none (loopback-only)")

    # stateless_http=True: no sticky sessions, matching the sessionless
    # protocol era and the guard-pattern elicitation design used above.
    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        path=args.path,
        stateless_http=True,
    )


# Export for use as module
__all__ = ["mcp", "main"]


if __name__ == "__main__":
    main()
