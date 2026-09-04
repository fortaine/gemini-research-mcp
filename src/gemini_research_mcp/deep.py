"""
Deep research using Gemini Deep Research Agent.

Provides comprehensive multi-step research with real-time progress.
Takes 3-20 minutes typically.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from google import genai

from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import (
    CLIENT_MAX_AGE_SECONDS,
    CLIENT_MAX_REQUESTS,
    DEFAULT_TIMEOUT,
    INITIAL_RETRY_BACKOFF,
    LOGGER_NAME,
    MAX_INITIAL_RETRIES,
    MAX_INITIAL_RETRY_DELAY,
    MAX_POLL_TIME,
    MAX_STREAM_RETRIES,
    MAX_STREAM_RETRY_DELAY,
    RECONNECT_DELAY,
    STREAM_POLL_INTERVAL,
    STREAM_RETRY_BACKOFF,
    get_api_key,
    get_deep_research_agent,
    get_model,
    is_retryable_error,
)
from gemini_research_mcp.types import (
    DeepResearchAgent,
    DeepResearchError,
    DeepResearchProgress,
    DeepResearchResult,
    DeepResearchUsage,
    ErrorCategory,
)

logger = logging.getLogger(LOGGER_NAME)

_GEMINI_UNSUPPORTED_MCP_SCHEMA_KEYWORDS = frozenset({
    "$ref",
    "$defs",
    "allOf",
    "anyOf",
    "dependencies",
    "dependentSchemas",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "then",
})
REMOTE_MCP_DISABLED_MESSAGE = (
    "Remote MCP is disabled for Deep Research and Deep Research Max because the "
    "provider does not reliably execute or retain MCP tool calls and results. "
    "Do not pass mcp_servers. You may use inspect_mcp_server_for_gemini for "
    "standalone endpoint diagnostics. Upstream issue: "
    "https://github.com/googleapis/python-genai/issues/2126"
)


def _validate_mcp_server_tool(server: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Gemini Interactions API MCP server tool."""
    url = server.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Each MCP server must include a non-empty 'url'.")
    if not url.startswith("https://"):
        allow_insecure = bool(server.get("allow_insecure_localhost"))
        is_local = url.startswith(("http://localhost", "http://127.0.0.1"))
        if not (allow_insecure and is_local):
            raise ValueError(
                "MCP server URLs must be HTTPS unless explicitly allowed for localhost."
            )

    tool: dict[str, Any] = {"type": "mcp_server", "url": url}
    name = server.get("name")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP server 'name' must be a non-empty string when provided.")
        tool["name"] = name

    headers = server.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ValueError("MCP server 'headers' must be a string-to-string object.")
        tool["headers"] = headers

    allowed_tools = server.get("allowed_tools")
    if allowed_tools is not None:
        if not isinstance(allowed_tools, list) or not all(
            isinstance(tool_name, str) and tool_name for tool_name in allowed_tools
        ):
            raise ValueError("MCP server 'allowed_tools' must be a list of tool names.")
        tool["allowed_tools"] = [{"tools": allowed_tools}]

    return tool


def _find_schema_keywords(schema: Any, keywords: frozenset[str], path: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            child_path = f"{path}.{key}"
            if key in keywords:
                matches.append(child_path)
            matches.extend(_find_schema_keywords(value, keywords, child_path))
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            matches.extend(_find_schema_keywords(item, keywords, f"{path}[{index}]"))
    return matches


def analyze_mcp_tool_for_gemini(tool: Mapping[str, Any]) -> list[str]:
    """Return likely Gemini Interactions compatibility issues for an MCP tool."""
    issues: list[str] = []

    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        issues.append("missing non-empty tool name")

    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        issues.append("missing tool description")

    input_schema = tool.get("inputSchema") or tool.get("input_schema")
    if not isinstance(input_schema, dict):
        issues.append("missing object input schema")
        return issues

    if input_schema.get("type") != "object":
        issues.append("input schema type should be object")

    properties = input_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        issues.append("input schema has no properties; add at least one explicit argument")

    unsupported = _find_schema_keywords(
        input_schema,
        _GEMINI_UNSUPPORTED_MCP_SCHEMA_KEYWORDS,
    )
    if unsupported:
        issues.append(
            "input schema uses keywords often rejected by Gemini: "
            + ", ".join(sorted(unsupported))
        )

    return issues


async def inspect_mcp_server_for_gemini(server: dict[str, Any]) -> dict[str, Any]:
    """Inspect a remote MCP server and summarize Gemini compatibility diagnostics."""
    normalized = _validate_mcp_server_tool(server)

    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    headers = normalized.get("headers")
    if not isinstance(headers, dict):
        headers = {}

    transport = StreamableHttpTransport(url=str(normalized["url"]), headers=headers)
    async with Client(transport=transport) as client:
        listed_tools = await client.list_tools()

    allowed_tools = server.get("allowed_tools")
    allowed_names = set(allowed_tools) if isinstance(allowed_tools, list) else set()

    tools: list[dict[str, Any]] = []
    discovered_names: set[str] = set()
    for listed_tool in listed_tools:
        tool = {
            "name": getattr(listed_tool, "name", None),
            "description": getattr(listed_tool, "description", None),
            "inputSchema": getattr(listed_tool, "inputSchema", None),
        }
        if isinstance(tool["name"], str):
            discovered_names.add(tool["name"])
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "allowed": tool["name"] in allowed_names if allowed_names else None,
            "input_schema": tool["inputSchema"],
            "issues": analyze_mcp_tool_for_gemini(tool),
        })

    server_issues: list[str] = []
    if not tools:
        server_issues.append("server returned no tools")
    missing_allowed_tools = sorted(allowed_names - discovered_names)
    if missing_allowed_tools:
        server_issues.append(
            "allowed_tools not found on server: " + ", ".join(missing_allowed_tools)
        )

    return {
        "server": {
            "name": normalized.get("name"),
            "url": normalized["url"],
            "allowed_tools": sorted(allowed_names) or None,
        },
        "issues": server_issues,
        "tools": tools,
    }


def validate_mcp_servers_supported(
    *,
    agent_name: DeepResearchAgent,
    mcp_servers: list[dict[str, Any]] | None,
) -> None:
    """Fail fast while provider-side Deep Research remote MCP is unreliable."""
    del agent_name
    if mcp_servers:
        raise ValueError(REMOTE_MCP_DISABLED_MESSAGE)


def build_interactions_tools(
    *,
    file_search_store_names: list[str] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]] | None:
    """Build Gemini Interactions API tools without logging or persisting secrets."""
    if mcp_servers:
        raise ValueError(REMOTE_MCP_DISABLED_MESSAGE)

    tools: list[dict[str, Any]] = []
    if file_search_store_names:
        tools.append({
            "type": "file_search",
            "file_search_store_names": file_search_store_names,
        })
    return tools or None


# =============================================================================
# Client Health Management
# =============================================================================


@dataclass
class ClientHealth:
    """Track client health for long-running servers."""

    created_at: float = field(default_factory=time.time)
    request_count: int = 0
    last_request_at: float = field(default_factory=time.time)
    consecutive_failures: int = 0

    def record_request(self) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.last_request_at = time.time()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1

    def needs_refresh(self) -> bool:
        """Check if client should be refreshed."""
        age = time.time() - self.created_at
        idle_time = time.time() - self.last_request_at

        # Refresh if client is too old
        if age > CLIENT_MAX_AGE_SECONDS:
            logger.info(
                "🔄 Client needs refresh: age=%.0fs > max=%.0fs",
                age, CLIENT_MAX_AGE_SECONDS
            )
            return True

        # Refresh if too many requests (if enabled)
        if CLIENT_MAX_REQUESTS > 0 and self.request_count >= CLIENT_MAX_REQUESTS:
            logger.info(
                "🔄 Client needs refresh: requests=%d >= max=%d",
                self.request_count, CLIENT_MAX_REQUESTS
            )
            return True

        # Refresh if too many consecutive failures
        if self.consecutive_failures >= 3:
            logger.info(
                "🔄 Client needs refresh: consecutive_failures=%d",
                self.consecutive_failures
            )
            return True

        # Refresh if idle for too long (half of max age)
        if idle_time > CLIENT_MAX_AGE_SECONDS / 2:
            logger.info("🔄 Client needs refresh: idle_time=%.0fs", idle_time)
            return True

        return False


# Global client management
_client: genai.Client | None = None
_client_health: ClientHealth | None = None


def _get_healthy_client() -> genai.Client:
    """Get a healthy Gemini client, creating a new one if needed."""
    global _client, _client_health

    if _client is None or _client_health is None or _client_health.needs_refresh():
        logger.info("🔌 Creating new Gemini client")
        _client = genai.Client(api_key=get_api_key())
        _client_health = ClientHealth()

    return _client


def _record_client_success() -> None:
    """Record a successful client operation."""
    global _client_health
    if _client_health:
        _client_health.record_request()


def _record_client_failure() -> None:
    """Record a failed client operation."""
    global _client_health
    if _client_health:
        _client_health.record_failure()


def _force_client_refresh() -> None:
    """Force client refresh on next request."""
    global _client, _client_health
    logger.warning("⚠️ Forcing client refresh due to critical failure")
    _client = None
    _client_health = None


def _get_interaction_field(value: Any, field_name: str) -> Any:
    """Read an Interactions SDK field from either a Pydantic model or dict."""
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _as_sequence(value: Any) -> list[Any]:
    """Normalize optional list-like SDK fields for safe iteration."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    ):
        return list(value)
    return []


def _is_reasoning_content(value: Any) -> bool:
    """Return whether a text content item represents non-report reasoning."""
    content_type = _get_interaction_field(value, "type")
    if content_type in {"thought", "thinking", "reasoning"}:
        return True
    for field_name in ("thought", "thinking", "reasoning"):
        if _get_interaction_field(value, field_name) is True:
            return True
    return False


def _is_image_content(value: Any) -> bool:
    """Return whether a content item is an image, so it is never merged into report text."""
    return bool(_get_interaction_field(value, "type") == "image")


def _extract_image_from_content(item: Any) -> dict[str, Any] | None:
    """Normalize an image content item (final, non-streaming interaction) to a plain dict."""
    if not _is_image_content(item):
        return None
    data = _get_interaction_field(item, "data")
    mime_type = _get_interaction_field(item, "mime_type")
    uri = _get_interaction_field(item, "uri")
    if not data and not uri:
        return None
    return {
        "data": data if isinstance(data, str) else None,
        "mime_type": str(mime_type) if mime_type else None,
        "uri": str(uri) if uri else None,
    }


def _extract_images_from_interaction(interaction: Any) -> list[dict[str, Any]]:
    """Extract image content items from a completed interaction's model_output steps."""
    images: list[dict[str, Any]] = []
    for step in _as_sequence(_get_interaction_field(interaction, "steps")):
        step_type = _get_interaction_field(step, "type")
        if step_type != "model_output":
            continue
        for item in _as_sequence(_get_interaction_field(step, "content")):
            image = _extract_image_from_content(item)
            if image is not None:
                images.append(image)
    return images


def _extract_usage(interaction: Any) -> DeepResearchUsage | None:
    """Extract usage/cost information from an interaction response."""
    usage_data = _get_interaction_field(interaction, "usage_metadata")

    if usage_data is None:
        usage_data = _get_interaction_field(interaction, "usage")

    if usage_data is None:
        return None

    prompt_tokens = _get_interaction_field(usage_data, "prompt_token_count")
    if prompt_tokens is None:
        prompt_tokens = _get_interaction_field(usage_data, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _get_interaction_field(usage_data, "total_input_tokens")

    completion_tokens = _get_interaction_field(usage_data, "candidates_token_count")
    if completion_tokens is None:
        completion_tokens = _get_interaction_field(usage_data, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _get_interaction_field(usage_data, "total_output_tokens")

    total_tokens = _get_interaction_field(usage_data, "total_token_count")
    if total_tokens is None:
        total_tokens = _get_interaction_field(usage_data, "total_tokens")

    raw_usage: dict[str, Any] = {}
    if hasattr(usage_data, "model_dump"):
        raw_usage = usage_data.model_dump(mode="json")
    elif isinstance(usage_data, dict):
        raw_usage = usage_data
    elif hasattr(usage_data, "__dict__"):
        raw_usage = vars(usage_data)
    elif hasattr(usage_data, "to_dict"):
        raw_usage = usage_data.to_dict()

    return DeepResearchUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage,
    )


def _extract_text_from_interaction(interaction: Any) -> str | None:
    """Extract the final text output from an interaction."""
    output_text = _get_interaction_field(interaction, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []

    for step in _as_sequence(_get_interaction_field(interaction, "steps")):
        step_type = _get_interaction_field(step, "type")
        if step_type != "model_output":
            continue
        for item in _as_sequence(_get_interaction_field(step, "content")):
            if _is_reasoning_content(item) or _is_image_content(item):
                continue
            text = _get_interaction_field(item, "text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)

    if text_parts:
        return "\n\n".join(text_parts)

    return None


def _get_stream_event_type(event: Any) -> str:
    """Return stream event type across google-genai interaction event shapes."""
    event_type = getattr(event, "event_type", None)
    if isinstance(event_type, str) and event_type:
        return event_type

    fallback_type = getattr(event, "type", None)
    if isinstance(fallback_type, str) and fallback_type:
        return fallback_type

    return "unknown"


def _extract_interaction_id(event: Any) -> str | None:
    """Extract interaction id across old and new interaction event shapes."""
    interaction = getattr(event, "interaction", None)
    if interaction is not None:
        if isinstance(interaction, dict):
            interaction_id = interaction.get("id")
        else:
            interaction_id = getattr(interaction, "id", None)
        if interaction_id:
            return str(interaction_id)

    event_interaction_id = getattr(event, "interaction_id", None)
    if event_interaction_id:
        return str(event_interaction_id)

    return None


def _extract_interaction_status(event: Any) -> str | None:
    """Extract interaction status across old and new interaction event shapes."""
    status = getattr(event, "status", None)
    if isinstance(status, str) and status:
        return status

    interaction = getattr(event, "interaction", None)
    if interaction is None:
        return None

    if isinstance(interaction, dict):
        interaction_status = interaction.get("status")
    else:
        interaction_status = getattr(interaction, "status", None)

    if isinstance(interaction_status, str) and interaction_status:
        return interaction_status
    return None


async def deep_research_stream(
    query: str,
    *,
    format_instructions: str | None = None,
    file_search_store_names: list[str] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    agent_name: DeepResearchAgent | None = None,
    visualization: str = "off",
    collaborative_planning: bool = False,
    previous_interaction_id: str | None = None,
) -> AsyncIterator[DeepResearchProgress]:
    """
    Stream deep research with real-time progress updates.

    Uses stream=True to receive thinking summaries and text deltas as they happen.
    Implements automatic reconnection on network interruptions with exponential backoff.

    Features:
    - Client health monitoring for long-running servers
    - Exponential backoff with configurable limits
    - Detailed logging for debugging null interaction_id issues
    - Automatic client refresh on consecutive failures

    Args:
        query: Research question or topic
        format_instructions: Optional formatting instructions for output
        file_search_store_names: Optional list of file search store names for RAG
        mcp_servers: Disabled compatibility parameter; non-empty values are rejected
        agent_name: Deep Research agent to use
        visualization: "off" (default) or "auto" - whether the Deep Research agent
            may include chart/diagram images in its response (agent_config.visualization).
        collaborative_planning: When True, the agent stops after drafting a research
            plan and waits for approval before executing it (agent_config.
            collaborative_planning). The stream ends with a "plan_ready" event
            instead of "complete" (Interactions API status "requires_action").
        previous_interaction_id: Continue a prior interaction (e.g. to refine or
            approve a previously-returned plan). See CreateAgentInteraction.

    Yields:
        DeepResearchProgress events with type:
        - "start": Research started, includes interaction_id
        - "thought": Thinking summary from the agent
        - "text": Text delta from the final report
        - "image": Image delta from the final report (never merged into text)
        - "complete": Research finished successfully
        - "plan_ready": Research plan is ready and awaiting approval
          (collaborative_planning=True only)
        - "error": Research failed
    """
    agent_name = agent_name or get_deep_research_agent()
    validate_mcp_servers_supported(agent_name=agent_name, mcp_servers=mcp_servers)

    prompt = f"{query}\n\n{format_instructions}" if format_instructions else query

    tools = build_interactions_tools(
        file_search_store_names=file_search_store_names,
        mcp_servers=mcp_servers,
    )

    create_kwargs: dict[str, Any] = {
        "input": prompt,
        "agent": agent_name,
        "background": True,
        "stream": True,
        "agent_config": {
            "type": "deep-research",
            "thinking_summaries": "auto",
            "visualization": visualization,
            "collaborative_planning": collaborative_planning,
        },
    }
    if tools:
        create_kwargs["tools"] = tools
    if previous_interaction_id:
        create_kwargs["previous_interaction_id"] = previous_interaction_id

    stream_start_time = time.time()

    logger.info("=" * 60)
    logger.info("🔬 DEEP RESEARCH AGENT")
    logger.info("   Agent: %s", agent_name)
    logger.info("   Query: %s", query[:100])
    logger.info("   Max initial retries: %d", MAX_INITIAL_RETRIES)
    logger.info("   Max stream retries: %d", MAX_STREAM_RETRIES)
    logger.info("=" * 60)

    interaction_id: str | None = None
    last_event_id: str | None = None
    is_complete = False
    initial_retry_delay = RECONNECT_DELAY
    stream_retry_delay = RECONNECT_DELAY
    stream_retry_count = 0
    disconnect_count = 0
    received_any_event = False

    async def process_stream(stream: Any) -> AsyncIterator[DeepResearchProgress]:
        """Process events from a stream (initial or resumed)."""
        nonlocal interaction_id, last_event_id, is_complete, received_any_event

        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            received_any_event = True
            elapsed = time.time() - stream_start_time

            chunk_type = _get_stream_event_type(chunk)
            logger.debug("[%.1fs] 📦 CHUNK #%d: type=%s", elapsed, chunk_count, chunk_type)

            if chunk_type in ("interaction.start", "interaction.created"):
                interaction_id = _extract_interaction_id(chunk)
                logger.info("[%.1fs] 🚀 %s: id=%s", elapsed, chunk_type, interaction_id)
                _record_client_success()  # Record successful API interaction
                yield DeepResearchProgress(
                    event_type="start",
                    interaction_id=interaction_id,
                    event_id=getattr(chunk, "event_id", None),
                )
                continue

            if hasattr(chunk, "event_id") and chunk.event_id:
                last_event_id = chunk.event_id

            if chunk_type in ("content.delta", "step.delta"):
                delta = getattr(chunk, "delta", None)
                if delta is None:
                    continue
                delta_type = (
                    delta.get("type")
                    if isinstance(delta, dict)
                    else getattr(delta, "type", None)
                )
                if delta_type == "thought_summary":
                    content = (
                        delta.get("content")
                        if isinstance(delta, dict)
                        else getattr(delta, "content", None)
                    )
                    if isinstance(content, dict):
                        thought_text = content.get("text") or str(content)
                    elif content is None:
                        thought_text = ""
                    else:
                        thought_text = content.text if hasattr(content, "text") else str(content)
                    logger.debug("[%.1fs] 🧠 thought_summary", elapsed)
                    yield DeepResearchProgress(
                        event_type="thought",
                        content=thought_text,
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif delta_type == "text":
                    if isinstance(delta, dict):
                        delta_text = delta.get("text")
                        content = delta.get("content")
                    else:
                        delta_text = getattr(delta, "text", None)
                        content = getattr(delta, "content", None)
                    if delta_text is None and content is not None:
                        if isinstance(content, dict):
                            delta_text = content.get("text")
                        else:
                            delta_text = getattr(content, "text", None)
                    logger.debug("[%.1fs] 📝 text delta: %d chars", elapsed, len(delta_text or ""))
                    yield DeepResearchProgress(
                        event_type="text",
                        content=delta_text,
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif delta_type == "image":
                    # Image deltas must never be folded into the text report - yield a
                    # dedicated "image" event carrying inline data or a hosted URI.
                    if isinstance(delta, dict):
                        image_data = delta.get("data")
                        image_mime_type = delta.get("mime_type")
                        image_uri = delta.get("uri")
                    else:
                        image_data = getattr(delta, "data", None)
                        image_mime_type = getattr(delta, "mime_type", None)
                        image_uri = getattr(delta, "uri", None)
                    logger.debug(
                        "[%.1fs] 🖼️ image delta: mime_type=%s, has_data=%s, has_uri=%s",
                        elapsed, image_mime_type, bool(image_data), bool(image_uri),
                    )
                    yield DeepResearchProgress(
                        event_type="image",
                        content=str(image_data) if image_data else None,
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                        image_mime_type=str(image_mime_type) if image_mime_type else None,
                        image_uri=str(image_uri) if image_uri else None,
                    )

            elif chunk_type in ("interaction.complete", "interaction.completed"):
                completed_interaction_id = _extract_interaction_id(chunk)
                if completed_interaction_id:
                    interaction_id = completed_interaction_id
                interaction_status = _extract_interaction_status(chunk) or "unknown"
                logger.info(
                    "[%.1fs] ✅ %s (status=%s)", elapsed, chunk_type, interaction_status
                )

                if interaction_status == "completed":
                    is_complete = True
                    yield DeepResearchProgress(
                        event_type="complete",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif interaction_status in ("cancelled", "canceled"):
                    is_complete = True
                    logger.warning(
                        "[%.1fs] 🚫 %s: cancelled",
                        elapsed,
                        chunk_type,
                    )
                    yield DeepResearchProgress(
                        event_type="error",
                        content="Research cancelled by provider.",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif interaction_status == "failed":
                    is_complete = True
                    logger.error(
                        "[%.1fs] ❌ %s: failed",
                        elapsed,
                        chunk_type,
                    )
                    yield DeepResearchProgress(
                        event_type="error",
                        content="Research failed on provider side.",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif interaction_status == "requires_action":
                    is_complete = True
                    logger.info(
                        "[%.1fs] 🗓️ interaction.completed: requires_action (plan ready)",
                        elapsed,
                    )
                    yield DeepResearchProgress(
                        event_type="plan_ready",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                else:
                    logger.warning(
                        "[%.1fs] ⚠️ %s but status='%s'",
                        elapsed,
                        chunk_type,
                        interaction_status,
                    )

            elif chunk_type in (
                "interaction.status_update",
                "interaction.in_progress",
                "interaction.requires_action",
                "interaction.failed",
            ):
                status_interaction_id = _extract_interaction_id(chunk)
                if status_interaction_id and interaction_id is None:
                    interaction_id = status_interaction_id

                status_update = _extract_interaction_status(chunk)
                if status_update is None and chunk_type.startswith("interaction."):
                    status_update = chunk_type.split(".", maxsplit=1)[1]

                logger.info(
                    "[%.1fs] ℹ️ %s (status=%s, id=%s)",
                    elapsed,
                    chunk_type,
                    status_update,
                    interaction_id,
                )

                if status_update in ("cancelled", "canceled"):
                    is_complete = True
                    yield DeepResearchProgress(
                        event_type="error",
                        content="Research cancelled by provider.",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif status_update in ("failed", "incomplete", "budget_exceeded"):
                    is_complete = True
                    yield DeepResearchProgress(
                        event_type="error",
                        content=f"Research failed on provider side (status={status_update}).",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif status_update == "requires_action":
                    is_complete = True
                    logger.info(
                        "[%.1fs] 🗓️ %s: requires_action (plan ready)", elapsed, chunk_type
                    )
                    yield DeepResearchProgress(
                        event_type="plan_ready",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif status_update == "completed":
                    is_complete = True
                    yield DeepResearchProgress(
                        event_type="complete",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )

            elif chunk_type == "error":
                is_complete = True
                error_msg = getattr(chunk, "error", "Unknown error")
                logger.error("[%.1fs] ❌ error: %s", elapsed, error_msg)
                yield DeepResearchProgress(
                    event_type="error",
                    content=str(error_msg),
                    interaction_id=interaction_id,
                    event_id=last_event_id,
                )

    # ==========================================================================
    # Phase 1: Initial connection with exponential backoff
    # ==========================================================================
    initial_attempt = 0

    while initial_attempt < MAX_INITIAL_RETRIES:
        initial_attempt += 1
        elapsed_t = time.time() - stream_start_time

        # Refresh client on each retry attempt to pick up health-based refreshes
        client = _get_healthy_client()

        try:
            logger.info(
                "⏱️ [%.1fs] 🔌 Initial connection attempt %d/%d...",
                elapsed_t, initial_attempt, MAX_INITIAL_RETRIES
            )

            stream = await client.aio.interactions.create(**create_kwargs)

            if stream is None:
                _record_client_failure()
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Stream returned None (attempt %d/%d)",
                    time.time() - stream_start_time, initial_attempt, MAX_INITIAL_RETRIES
                )
                # Exponential backoff for retries
                backoff = INITIAL_RETRY_BACKOFF ** (initial_attempt - 1)
                wait_time = min(initial_retry_delay * backoff, MAX_INITIAL_RETRY_DELAY)
                logger.info("   ⏳ Waiting %.1fs before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue

            logger.info("⏱️ [%.1fs] ✅ Stream connected", time.time() - stream_start_time)
            async for progress in process_stream(stream):
                yield progress

            # If we got here without receiving an interaction start event, log it
            if interaction_id is None and received_any_event:
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Stream ended but never received interaction start event",
                    time.time() - stream_start_time
                )
            break

        except TypeError as e:
            if "NoneType" in str(e) and "not iterable" in str(e):
                _record_client_failure()
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Stream returned None (TypeError, attempt %d/%d): %s",
                    time.time() - stream_start_time, initial_attempt, MAX_INITIAL_RETRIES, e
                )
                backoff = INITIAL_RETRY_BACKOFF ** (initial_attempt - 1)
                wait_time = min(initial_retry_delay * backoff, MAX_INITIAL_RETRY_DELAY)
                logger.info("   ⏳ Waiting %.1fs before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            disconnect_count += 1
            _record_client_failure()
            elapsed_t = time.time() - stream_start_time
            logger.warning(
                "⏱️ [%.1fs] ❌ DISCONNECT #%d (TypeError): %s",
                elapsed_t, disconnect_count, e
            )
            break

        except Exception as e:
            disconnect_count += 1
            _record_client_failure()
            elapsed_t = time.time() - stream_start_time
            error_str = str(e)
            logger.warning(
                "⏱️ [%.1fs] ❌ DISCONNECT #%d: %s",
                elapsed_t, disconnect_count, error_str
            )

            # Check if this is a retryable error
            if is_retryable_error(error_str) and initial_attempt < MAX_INITIAL_RETRIES:
                backoff = INITIAL_RETRY_BACKOFF ** (initial_attempt - 1)
                wait_time = min(initial_retry_delay * backoff, MAX_INITIAL_RETRY_DELAY)
                logger.info("   🔄 Retryable error, waiting %.1fs before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            break

    # ==========================================================================
    # Phase 2: Check if we have interaction_id for reconnection
    # ==========================================================================
    if interaction_id is None and not is_complete:
        elapsed = time.time() - stream_start_time
        logger.error(
            "⏱️ [%.1fs] ❌ CRITICAL: No interaction_id received after %d initial attempts. "
            "This may indicate API issues or rate limiting. "
            "Server restart may be required if this persists.",
            elapsed, initial_attempt
        )
        # Force client refresh for next request
        _force_client_refresh()
        yield DeepResearchProgress(
            event_type="error",
            content=(
                f"Failed to start research after {initial_attempt} attempts ({elapsed:.0f}s). "
                f"No interaction_id received from API. This may be a temporary API issue. "
                f"Please try again in a few minutes."
            ),
            interaction_id=None,
            event_id=None,
        )
        return

    # ==========================================================================
    # Phase 3: Reconnection loop with exponential backoff (if stream interrupted)
    # ==========================================================================
    while not is_complete and interaction_id and stream_retry_count < MAX_STREAM_RETRIES:
        stream_retry_count += 1
        elapsed = time.time() - stream_start_time
        short_id = interaction_id[:16] + "..." if len(interaction_id) > 16 else interaction_id
        logger.info(
            "⏱️ [%.1fs] 🔄 RECONNECT attempt %d/%d (id=%s)",
            elapsed, stream_retry_count, MAX_STREAM_RETRIES, short_id
        )

        # Exponential backoff
        backoff = STREAM_RETRY_BACKOFF ** (stream_retry_count - 1)
        wait_time = min(stream_retry_delay * backoff, MAX_STREAM_RETRY_DELAY)
        logger.info("   ⏳ Waiting %.1fs before reconnect...", wait_time)
        await asyncio.sleep(wait_time)

        try:
            # Refresh client if needed before reconnection attempt
            client = _get_healthy_client()

            # last_event_id can be None on first reconnect
            get_kwargs: dict[str, Any] = {"id": interaction_id, "stream": True}
            if last_event_id is not None:
                get_kwargs["last_event_id"] = last_event_id

            resume_stream = await client.aio.interactions.get(**get_kwargs)

            # Validate stream before recording success (API may return None)
            if resume_stream is None:
                _record_client_failure()
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Reconnect returned None (attempt %d/%d)",
                    time.time() - stream_start_time, stream_retry_count, MAX_STREAM_RETRIES
                )
                continue

            logger.info(
                "⏱️ [%.1fs] ✅ RECONNECTED successfully",
                time.time() - stream_start_time
            )
            _record_client_success()

            async for progress in process_stream(resume_stream):
                yield progress
                # Reset retry count on successful event
                stream_retry_count = 0

        except Exception as e:
            disconnect_count += 1
            _record_client_failure()
            elapsed_t = time.time() - stream_start_time
            error_str = str(e)
            logger.warning(
                "⏱️ [%.1fs] ❌ RECONNECT FAILED #%d: %s",
                elapsed_t, disconnect_count, error_str
            )

            # Force client refresh after multiple failures
            if disconnect_count >= 3:
                _force_client_refresh()

    # ==========================================================================
    # Phase 4: Final status check
    # ==========================================================================
    if not is_complete:
        elapsed = time.time() - stream_start_time
        logger.error(
            "⏱️ [%.1fs] ❌ RESEARCH FAILED: disconnects=%d, retries=%d, id=%s",
            elapsed, disconnect_count, stream_retry_count, interaction_id
        )
        yield DeepResearchProgress(
            event_type="error",
            content=(
                f"Research interrupted after {elapsed:.0f}s "
                f"({disconnect_count} disconnects, {stream_retry_count} reconnect attempts). "
                f"Interaction ID: {interaction_id}"
            ),
            interaction_id=interaction_id,
            event_id=last_event_id,
        )


async def deep_research(
    query: str,
    *,
    format_instructions: str | None = None,
    file_search_store_names: list[str] | None = None,
    on_progress: Callable[[DeepResearchProgress], None | Awaitable[None]] | None = None,
    agent_name: DeepResearchAgent | None = None,
    resolve_citations: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> DeepResearchResult:
    """
    Comprehensive multi-step research using Gemini Deep Research Agent.

    Uses streaming internally to receive real-time thinking summaries
    and progress updates. The agent autonomously plans, searches, reads,
    and synthesizes information to produce a detailed report.

    Takes 3-20 minutes typically.

    Args:
        query: Research question or topic
        format_instructions: Optional formatting instructions for output
        file_search_store_names: Optional list of file search store names for RAG
        on_progress: Callback for each progress event (sync or async)
        agent_name: Deep Research agent to use
        resolve_citations: Whether to extract and resolve citation URLs
        timeout: Maximum wait time in seconds

    Returns:
        DeepResearchResult with collected text, thinking summaries, usage, and citations

    Raises:
        DeepResearchError: On timeout, failure, or API errors
    """
    start_time = time.time()
    text_parts: list[str] = []
    thinking_summaries: list[str] = []
    interaction_id: str | None = None
    raw_interaction: Any = None

    async for progress in deep_research_stream(
        query,
        format_instructions=format_instructions,
        file_search_store_names=file_search_store_names,
        agent_name=agent_name,
    ):
        if on_progress:
            cb_result = on_progress(progress)
            if inspect.isawaitable(cb_result):
                await cb_result

        if progress.event_type == "start":
            interaction_id = progress.interaction_id
        elif progress.event_type == "thought":
            if progress.content:
                thinking_summaries.append(progress.content)
        elif progress.event_type == "text":
            if progress.content:
                text_parts.append(progress.content)
        elif progress.event_type == "error":
            raise DeepResearchError(
                code="RESEARCH_FAILED",
                message=f"Deep Research failed: {progress.content}",
                details={"interaction_id": interaction_id},
            )

    final_text = "".join(text_parts)

    # Post-stream polling if we got no text but have interaction_id
    if not final_text.strip() and interaction_id:
        logger.info("🔄 POLLING: Stream ended without text...")
        client = _get_healthy_client()  # Use health-monitored client
        poll_start = time.time()

        while time.time() - poll_start < MAX_POLL_TIME:
            try:
                final_interaction = await client.aio.interactions.get(id=interaction_id)
                _record_client_success()  # Keep client alive during polling
                status = getattr(final_interaction, "status", "unknown")

                if on_progress:
                    elapsed = time.time() - poll_start
                    prog = DeepResearchProgress(
                        event_type="status",
                        content=f"Waiting... ({status}, {elapsed:.0f}s)",
                        interaction_id=interaction_id,
                    )
                    poll_cb_result = on_progress(prog)
                    if inspect.isawaitable(poll_cb_result):
                        await poll_cb_result

                if status == "completed":
                    raw_interaction = final_interaction
                    final_text = _extract_text_from_interaction(final_interaction) or ""
                    break

                elif status in ("cancelled", "canceled"):
                    raise DeepResearchError(
                        code="RESEARCH_CANCELLED",
                        message="Research cancelled by provider.",
                        details={"interaction_id": interaction_id},
                        category=ErrorCategory.RESEARCH_CANCELLED,
                    )

                elif status == "failed":
                    error = getattr(final_interaction, "error", "Unknown error")
                    raise DeepResearchError(
                        code="RESEARCH_FAILED",
                        message=str(error),
                        details={"interaction_id": interaction_id},
                    )

                await asyncio.sleep(STREAM_POLL_INTERVAL)

            except DeepResearchError:
                raise
            except Exception as e:
                if is_retryable_error(str(e)):
                    await asyncio.sleep(STREAM_POLL_INTERVAL)
                else:
                    raise

        if not final_text.strip():
            raise DeepResearchError(
                code="TIMEOUT",
                message="Deep Research timed out",
                details={"interaction_id": interaction_id},
            )

    duration_seconds = time.time() - start_time
    usage = _extract_usage(raw_interaction) if raw_interaction else None

    result = DeepResearchResult(
        text=final_text,
        citations=[],
        thinking_summaries=thinking_summaries,
        interaction_id=interaction_id,
        usage=usage,
        duration_seconds=duration_seconds,
        raw_interaction=raw_interaction,
    )

    if resolve_citations and final_text:
        result = await process_citations(result, resolve_urls=True)

    return result


async def get_research_status(interaction_id: str) -> DeepResearchResult:
    """
    Get the current status of a Deep Research task.

    Internal helper used by research_deep to poll for completion.

    Args:
        interaction_id: The interaction ID from a research task

    Returns:
        DeepResearchResult with current status and any available report text
    """
    client = _get_healthy_client()  # Use health-monitored client
    interaction = await client.aio.interactions.get(id=interaction_id)
    _record_client_success()

    status = getattr(interaction, "status", "unknown")
    text = _extract_text_from_interaction(interaction) if status == "completed" else None
    usage = _extract_usage(interaction)
    images = _extract_images_from_interaction(interaction) if status == "completed" else []

    return DeepResearchResult(
        text=text or "",
        citations=[],
        thinking_summaries=[],
        interaction_id=interaction_id,
        usage=usage,
        raw_interaction=interaction,
        images=images,
    )


async def research_followup(
    previous_interaction_id: str,
    query: str,
    *,
    model: str | None = None,
) -> str:
    """
    Ask a follow-up question about a completed Deep Research task.

    This continues the conversation context from a previous research task,
    allowing clarification, summarization, or elaboration on specific sections
    without restarting the entire research.

    Args:
        previous_interaction_id: Interaction ID from a completed research task
                                 (available as result.interaction_id from research_deep)
        query: The follow-up question
        model: Model to use for the follow-up. Defaults to the configured
               GEMINI_MODEL / DEFAULT_MODEL (currently gemini-3.8-flash).

    Returns:
        The text response to the follow-up question

    Raises:
        DeepResearchError: On invalid interaction ID or API errors
    """
    model = model or get_model()
    logger.info("💬 Follow-up question for %s: %s", previous_interaction_id, query[:100])

    client = _get_healthy_client()  # Use health-monitored client

    try:
        interaction = await client.aio.interactions.create(
            input=query,
            model=model,
            previous_interaction_id=previous_interaction_id,
        )
        _record_client_success()

        # Extract text from the response
        text = _extract_text_from_interaction(interaction)

        if not text:
            raise DeepResearchError(
                code="NO_RESPONSE",
                message="No response received from follow-up",
                details={"previous_interaction_id": previous_interaction_id},
            )

        logger.info("   ✅ Follow-up response received")
        return text

    except Exception as e:
        logger.exception("Follow-up question failed: %s", e)
        raise DeepResearchError(
            code="FOLLOWUP_FAILED",
            message=str(e),
            details={"previous_interaction_id": previous_interaction_id},
        ) from e
