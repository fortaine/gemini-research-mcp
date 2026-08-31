"""
Configuration management for Deep Research MCP Server.

All configuration is loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from gemini_research_mcp.types import DeepResearchAgent

# =============================================================================
# Logging
# =============================================================================

LOGGER_NAME = "gemini-research-mcp"

# Distributed backends. Local defaults remain DiskStore for application data
# and FastMCP's memory backend for Tasks.
STORAGE_URL_ENV_VAR = "GEMINI_RESEARCH_STORAGE_URL"
DOCKET_URL_ENV_VAR = "FASTMCP_DOCKET_URL"


# =============================================================================
# Model Configuration
# =============================================================================

# Default models - can be overridden via environment
DEFAULT_MODEL = "gemini-3.7-flash"
# Interactions Deep Research agent used by research_deep by default.
DEFAULT_DEEP_RESEARCH_AGENT = DeepResearchAgent.DEEP_RESEARCH
# Model for generating summaries (fast, cheap)
DEFAULT_SUMMARY_MODEL = "gemini-3.7-flash"

# Thinking level for Gemini 3 models
# Values: "minimal", "low", "medium", "high"
# - minimal: minimize latency for chat/high-throughput (gemini-3.1-* only;
#   gemini-3.7-flash does not document "minimal", use "low" instead)
# - low: balance speed and quality
# - medium: good reasoning depth
# - high: maximum reasoning depth (recommended for research)
DEFAULT_THINKING_LEVEL = "high"


# =============================================================================
# Timeouts and Polling
# =============================================================================

# Deep Research configuration
STREAM_POLL_INTERVAL = 10.0  # seconds between polls
MAX_POLL_TIME = 3600.0  # 60 minutes max wait
DEFAULT_TIMEOUT = 3600.0  # 60 minutes default timeout
RECONNECT_DELAY = 2.0  # Initial delay before reconnection
MAX_INITIAL_RETRIES = 5  # Retries for initial stream creation (increased for reliability)
MAX_INITIAL_RETRY_DELAY = 30.0  # Maximum delay between initial retries
INITIAL_RETRY_BACKOFF = 1.5  # Exponential backoff multiplier for initial retries

# Reconnection configuration for stream interruptions
MAX_STREAM_RETRIES = 10  # Maximum reconnection attempts after stream established
MAX_STREAM_RETRY_DELAY = 60.0  # Maximum delay between stream reconnection attempts
STREAM_RETRY_BACKOFF = 1.5  # Exponential backoff multiplier for stream retries

# Client health monitoring (for long-running servers)
# Refresh triggers: age > max, requests >= max, failures >= 3, or idle > max/2
CLIENT_MAX_AGE_SECONDS = 3600.0  # Max client age (1 hour); also refreshes if idle > 30min
CLIENT_MAX_REQUESTS = 100  # Recreate client after N requests (0 = disabled)

# Errors that should trigger reconnection
RETRYABLE_ERRORS = [
    "gateway_timeout",
    "deadline_expired",
    "timeout",
    "connection_reset",
    "closed",
    "aborted",
    "internal_error",
    "service_unavailable",
]


# =============================================================================
# Getters
# =============================================================================


def get_api_key() -> str:
    """Get Gemini API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return api_key


def get_model() -> str:
    """Get model name with env override support."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def get_deep_research_agent() -> DeepResearchAgent:
    """Get Deep Research agent.

    Environment variable aliases:
    - fast, standard, deep-research -> Deep Research
    - max, deep-research-max -> Deep Research Max
    - pro, deep-research-pro -> legacy December 2025 agent
    """
    raw = os.environ.get("DEEP_RESEARCH_AGENT")
    if not raw:
        return DEFAULT_DEEP_RESEARCH_AGENT
    normalized = raw.strip().lower()
    aliases = {
        "fast": DeepResearchAgent.DEEP_RESEARCH,
        "standard": DeepResearchAgent.DEEP_RESEARCH,
        "deep-research": DeepResearchAgent.DEEP_RESEARCH,
        DeepResearchAgent.DEEP_RESEARCH.value: DeepResearchAgent.DEEP_RESEARCH,
        "max": DeepResearchAgent.DEEP_RESEARCH_MAX,
        "deep-research-max": DeepResearchAgent.DEEP_RESEARCH_MAX,
        DeepResearchAgent.DEEP_RESEARCH_MAX.value: DeepResearchAgent.DEEP_RESEARCH_MAX,
        "pro": DeepResearchAgent.DEEP_RESEARCH_PRO,
        "deep-research-pro": DeepResearchAgent.DEEP_RESEARCH_PRO,
        DeepResearchAgent.DEEP_RESEARCH_PRO.value: DeepResearchAgent.DEEP_RESEARCH_PRO,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Invalid DEEP_RESEARCH_AGENT '{raw}'. Use one of: {valid}") from exc


def get_summary_model() -> str:
    """Get model for generating summaries (fast, cheap)."""
    return os.environ.get("GEMINI_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL)


def get_tasks_backend_url() -> str | None:
    """Resolve the Tasks backend without mutating process-wide environment.

    An explicit FASTMCP_DOCKET_URL always wins. Otherwise a configured
    Redis-compatible application storage URL is reused so one URL can share
    sessions, exports, and Tasks across workers. With neither value, returning
    None preserves FastMCP's local memory default.
    """
    explicit = os.environ.get(DOCKET_URL_ENV_VAR)
    if explicit:
        return explicit

    shared = os.environ.get(STORAGE_URL_ENV_VAR)
    if shared and urlsplit(shared).scheme.lower() in {"redis", "rediss"}:
        return shared
    return None


# =============================================================================
# Export directory
# =============================================================================

DEFAULT_EXPORT_DIR = Path.home() / ".gemini-research" / "exports"


def get_export_dir() -> Path:
    """Return the directory where exports are written when no output_path is supplied.

    Resolution order:
    1. ``GEMINI_RESEARCH_EXPORT_DIR`` environment variable.
    2. ``~/.gemini-research/exports/``.
    3. System temp directory (fallback if neither above is writable).

    The returned directory is guaranteed to exist and be writable.
    """
    candidates: list[Path] = []
    custom = os.environ.get("GEMINI_RESEARCH_EXPORT_DIR")
    if custom:
        candidates.append(Path(custom).expanduser())
    candidates.append(DEFAULT_EXPORT_DIR)
    candidates.append(Path(tempfile.gettempdir()) / "gemini-research-exports")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Confirm writability without leaving a file behind
            probe = candidate / ".write-probe"
            probe.touch()
            probe.unlink()
            return candidate.resolve()
        except OSError:
            continue

    # Last resort: system temp (guaranteed writable on all supported platforms)
    fallback = Path(tempfile.gettempdir())
    return fallback.resolve()


def is_retryable_error(error_msg: str) -> bool:
    """Check if an error message indicates a retryable condition."""
    error_lower = str(error_msg).lower()
    return any(err in error_lower for err in RETRYABLE_ERRORS)


def default_system_prompt() -> str:
    """Default system prompt for research tasks."""
    today = date.today().strftime("%B %d, %Y")
    return f"""You are an expert research analyst. Today is {today}.

When answering questions:
1. Provide accurate, well-researched information grounded in your search results
2. Cite your sources by referring to the specific information from each source
3. If information is uncertain or sources conflict, acknowledge this
4. Structure complex answers with clear headings and bullet points
5. Prioritize recent and authoritative sources

Your goal is to provide comprehensive, factual answers that would satisfy
a professional researcher."""
