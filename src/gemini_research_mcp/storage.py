"""
Persistent storage for research sessions.

Stores interaction metadata for later retrieval and follow-up conversations.
Uses py-key-value-aio DiskStore for persistent storage with automatic TTL cleanup.

Storage Location (XDG-compliant via platformdirs):
- macOS: ~/Library/Application Support/gemini-research-mcp/
- Linux: ~/.local/share/gemini-research-mcp/
- Windows: %APPDATA%\\gemini-research-mcp\\

Gemini Interaction Retention:
- Paid tier: 55 days
- Free tier: 24 hours
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import platformdirs
from diskcache import Cache  # type: ignore[import-untyped]
from key_value.aio.protocols.key_value import AsyncEnumerateKeysProtocol
from key_value.aio.stores.base import BaseStore
from key_value.aio.stores.disk import DiskStore

from gemini_research_mcp.config import LOGGER_NAME, STORAGE_URL_ENV_VAR
from gemini_research_mcp.types import DeepResearchAgent

logger = logging.getLogger(LOGGER_NAME)

T = TypeVar("T")

# =============================================================================
# Configuration
# =============================================================================

# Default TTL matches Gemini paid tier (55 days in seconds)
DEFAULT_TTL_SECONDS = 55 * 24 * 60 * 60  # 55 days

# Free tier TTL (24 hours)
FREE_TIER_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Application name for platformdirs
APP_NAME = "gemini-research-mcp"

# Storage collection names (namespaces within the key-value backend)
SESSIONS_COLLECTION = "sessions"
EXPORTS_COLLECTION = "exports"

# Versioned sidecar index used only by DiskStore, which has no public keys()
# API. Redis and other enumerable backends use their public protocol directly.
DISK_INDEX_FILENAME = ".gemini-research-key-index-v1.json"
DISK_INDEX_VERSION = 1
_INDEXED_COLLECTIONS = (SESSIONS_COLLECTION, EXPORTS_COLLECTION)
_DISK_INDEX_LOCKS: dict[Path, threading.Lock] = {}
_DISK_INDEX_LOCKS_GUARD = threading.Lock()


def get_storage_backend_url() -> str | None:
    """Return the configured shared-storage URL, or None for the local disk default.

    Set GEMINI_RESEARCH_STORAGE_URL to a redis:// URL (requires the
    `distributed` extra: `pip install gemini-research-mcp[distributed]`) to
    make sessions and export artifacts shared across multiple server
    instances/workers instead of being local to one process's disk.
    """
    return os.environ.get(STORAGE_URL_ENV_VAR) or None


def get_storage_dir() -> Path:
    """Get storage directory from env or XDG-compliant default.

    If GEMINI_RESEARCH_STORAGE_PATH is set:
    - If it's a directory path (exists or ends with /), use it directly
    - Otherwise, treat it as a file path and use its parent
    """
    custom_path = os.environ.get("GEMINI_RESEARCH_STORAGE_PATH")
    if custom_path:
        # Expand ~ and resolve path
        expanded = Path(custom_path).expanduser().resolve()
        # If path exists and is a directory, or ends with /, use it directly
        if expanded.is_dir() or custom_path.endswith(os.sep) or custom_path.endswith("/"):
            return expanded
        # Otherwise treat as file path and use parent
        return expanded.parent
    return Path(platformdirs.user_data_dir(APP_NAME))


def create_store(*, storage_dir: Path | None = None) -> BaseStore:
    """Create the key-value backend: shared Redis when configured, else local disk.

    This is the single place that decides between backends, so both
    SessionStorage and ExportArtifactStore share identical selection logic
    and both become shareable across processes/instances the moment
    GEMINI_RESEARCH_STORAGE_URL is set - no other code needs to know which
    backend is active.
    """
    url = get_storage_backend_url()
    if url is not None:
        # Imported lazily: redis is an optional dependency (the `distributed`
        # extra) and must not be required for the default local-disk mode.
        from key_value.aio.stores.redis.store import RedisStore

        logger.info("💾 Using shared storage backend (Redis) from %s", STORAGE_URL_ENV_VAR)
        return RedisStore(url=url)

    directory = storage_dir or get_storage_dir()
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug("💾 Using local disk storage backend at %s", directory)
    return DiskStore(directory=str(directory))


def get_ttl_seconds() -> int:
    """Get TTL from env or default (55 days for paid tier)."""
    custom_ttl = os.environ.get("GEMINI_RESEARCH_TTL_SECONDS")
    if custom_ttl:
        try:
            return int(custom_ttl)
        except ValueError:
            logger.warning("Invalid GEMINI_RESEARCH_TTL_SECONDS: %s", custom_ttl)
    return DEFAULT_TTL_SECONDS


class _DiskKeyIndex:
    """Project-owned key index for the local DiskStore backend."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.path = storage_dir / DISK_INDEX_FILENAME
        self._lock = asyncio.Lock()
        resolved_path = self.path.resolve()
        with _DISK_INDEX_LOCKS_GUARD:
            self._file_lock = _DISK_INDEX_LOCKS.setdefault(
                resolved_path, threading.Lock()
            )

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": DISK_INDEX_VERSION,
            "collections": {collection: [] for collection in _INDEXED_COLLECTIONS},
        }

    def _validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or data.get("version") != DISK_INDEX_VERSION:
            raise ValueError("unsupported disk index version")

        collections = data.get("collections")
        if not isinstance(collections, dict):
            raise ValueError("disk index collections must be an object")

        normalized = self._empty()
        for collection in _INDEXED_COLLECTIONS:
            keys = collections.get(collection, [])
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise ValueError(f"disk index collection {collection!r} is invalid")
            normalized["collections"][collection] = sorted(set(keys))
        return normalized

    def _scan_cache(self) -> dict[str, Any]:
        data = self._empty()
        cache = Cache(directory=str(self.storage_dir), eviction_policy="none")
        try:
            for raw_key in cache.iterkeys():
                if not isinstance(raw_key, str) or cache.get(raw_key) is None:
                    continue
                for collection in _INDEXED_COLLECTIONS:
                    prefix = f"{collection}::"
                    if raw_key.startswith(prefix):
                        data["collections"][collection].append(raw_key[len(prefix):])
                        break
        finally:
            cache.close()

        for collection in _INDEXED_COLLECTIONS:
            data["collections"][collection] = sorted(
                set(data["collections"][collection])
            )
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.{id(self)}.tmp"
        )
        try:
            temp_path.write_text(
                json.dumps(data, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temp_path, self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load_or_rebuild(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return self._validate(raw)
        except FileNotFoundError:
            logger.info("Building local storage key index at %s", self.path)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Rebuilding invalid local storage key index: %s", exc)

        data = self._scan_cache()
        self._write(data)
        return data

    def _keys_sync(self, collection: str) -> list[str]:
        with self._file_lock:
            data = self._load_or_rebuild()
            return list(data["collections"][collection])

    def _add_sync(self, collection: str, key: str) -> None:
        with self._file_lock:
            data = self._load_or_rebuild()
            keys = set(data["collections"][collection])
            keys.add(key)
            data["collections"][collection] = sorted(keys)
            self._write(data)

    def _discard_many_sync(
        self, collection: str, keys_to_remove: list[str]
    ) -> None:
        with self._file_lock:
            data = self._load_or_rebuild()
            keys = set(data["collections"][collection])
            keys.difference_update(keys_to_remove)
            data["collections"][collection] = sorted(keys)
            self._write(data)

    async def keys(self, collection: str) -> list[str]:
        async with self._lock:
            return await asyncio.to_thread(self._keys_sync, collection)

    async def add(self, collection: str, key: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._add_sync, collection, key)

    async def discard_many(self, collection: str, keys_to_remove: list[str]) -> None:
        if not keys_to_remove:
            return
        async with self._lock:
            await asyncio.to_thread(
                self._discard_many_sync, collection, keys_to_remove
            )


async def _enumerate_collection_keys(
    store: BaseStore,
    *,
    collection: str,
    disk_index: _DiskKeyIndex | None,
) -> list[str]:
    if isinstance(store, AsyncEnumerateKeysProtocol):
        return await store.keys(collection=collection)
    if disk_index is not None:
        return await disk_index.keys(collection)
    raise RuntimeError(
        f"Storage backend {type(store).__name__} does not support key enumeration."
    )


# =============================================================================
# Data Types
# =============================================================================


class ResearchStatus(StrEnum):
    """Status of a research session for resume functionality."""

    IN_PROGRESS = "in_progress"  # Research started, not yet completed
    COMPLETED = "completed"  # Research finished successfully
    FAILED = "failed"  # Research failed with error
    INTERRUPTED = "interrupted"  # Research interrupted (VS Code disconnected, etc.)
    CANCELLED = "cancelled"  # Research cancelled by provider or user
    # Collaborative planning (agent_config.collaborative_planning=True) statuses.
    # These sit between IN_PROGRESS and COMPLETED/FAILED/... in the lifecycle:
    # PLANNING -> AWAITING_APPROVAL -> EXECUTING -> COMPLETED (or FAILED/CANCELLED).
    PLANNING = "planning"  # Agent is drafting the research plan
    AWAITING_APPROVAL = "awaiting_approval"  # Plan delivered; needs refine/approve
    EXECUTING = "executing"  # Plan approved; final report is being generated


@dataclass
class ResearchSession:
    """A stored research session with metadata."""

    interaction_id: str
    query: str
    created_at: float  # Unix timestamp
    title: str | None = None  # Short descriptive title
    summary: str | None = None  # AI-generated synopsis for discovery
    report_text: str | None = None  # Full research report
    format_instructions: str | None = None
    agent_name: DeepResearchAgent | None = None
    duration_seconds: float | None = None
    total_tokens: int | None = None
    expires_at: float | None = None  # Unix timestamp
    tags: list[str] = field(default_factory=list)
    notes: str | None = None  # User-added notes
    status: ResearchStatus = ResearchStatus.COMPLETED  # For resume functionality
    plan_text: str | None = None  # Draft plan while PLANNING/AWAITING_APPROVAL
    image_export_ids: list[str] = field(default_factory=list)  # Persisted image artifacts

    def __post_init__(self) -> None:
        """Set expiration if not provided."""
        if self.expires_at is None:
            self.expires_at = self.created_at + get_ttl_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def created_at_iso(self) -> str:
        """Return created_at as ISO format string."""
        return datetime.fromtimestamp(self.created_at, tz=UTC).isoformat()

    @property
    def expires_at_iso(self) -> str | None:
        """Return expires_at as ISO format string."""
        if self.expires_at is None:
            return None
        return datetime.fromtimestamp(self.expires_at, tz=UTC).isoformat()

    @property
    def time_remaining(self) -> float | None:
        """Seconds remaining until expiration."""
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - time.time())

    @property
    def time_remaining_human(self) -> str | None:
        """Human-readable time remaining."""
        remaining = self.time_remaining
        if remaining is None:
            return None
        if remaining <= 0:
            return "expired"

        days = int(remaining // (24 * 60 * 60))
        hours = int((remaining % (24 * 60 * 60)) // (60 * 60))

        if days > 0:
            return f"{days}d {hours}h"
        minutes = int((remaining % (60 * 60)) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to string for JSON serialization
        result["status"] = self.status.value
        if self.agent_name is not None:
            result["agent_name"] = self.agent_name.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchSession:
        """Create from dictionary.

        Handles missing optional fields gracefully.

        Raises:
            KeyError: If required fields (interaction_id, query, created_at) are missing.
        """
        # Validate required fields explicitly for better error messages
        required = ["interaction_id", "query", "created_at"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required fields: {', '.join(missing)}")

        # Handle agent_name enum
        agent_name_raw = data.get("agent_name")
        agent_name = (
            DeepResearchAgent(agent_name_raw) if agent_name_raw is not None else None
        )

        return cls(
            interaction_id=data["interaction_id"],
            query=data["query"],
            created_at=data["created_at"],
            title=data.get("title"),
            summary=data.get("summary"),
            report_text=data.get("report_text"),
            format_instructions=data.get("format_instructions"),
            agent_name=agent_name,
            duration_seconds=data.get("duration_seconds"),
            total_tokens=data.get("total_tokens"),
            expires_at=data.get("expires_at"),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            status=ResearchStatus(data.get("status", "completed")),
            plan_text=data.get("plan_text"),
            image_export_ids=data.get("image_export_ids", []),
        )

    @property
    def is_resumable(self) -> bool:
        """Check if the session can be resumed (in_progress or interrupted)."""
        return self.status in (ResearchStatus.IN_PROGRESS, ResearchStatus.INTERRUPTED)

    def short_description(self) -> str:
        """Return a short description for listing."""
        title = self.title or self.query[:50]
        remaining = self.time_remaining_human or "unknown"
        return f"[{self.interaction_id[:12]}...] {title} (expires: {remaining})"


# =============================================================================
# Storage Operations
# =============================================================================


class SessionStorage:
    """
    Persistent storage for research sessions using py-key-value-aio DiskStore.

    Features:
    - XDG-compliant storage paths via platformdirs
    - Automatic TTL-based expiration (handled by diskcache)
    - Async-first API with sync wrappers for convenience
    - Industry-standard key-value interface (used by FastMCP, agentpool, etc.)
    """

    def __init__(self, storage_dir: Path | None = None):
        """Initialize storage, using a shared backend if GEMINI_RESEARCH_STORAGE_URL is set."""
        self.storage_dir = storage_dir or get_storage_dir()
        self._store = create_store(storage_dir=self.storage_dir)
        self._disk_index = (
            _DiskKeyIndex(self.storage_dir) if isinstance(self._store, DiskStore) else None
        )
        logger.debug("💾 Storage initialized at %s", self.storage_dir)

    # -------------------------------------------------------------------------
    # Key Enumeration (backend-agnostic)
    # -------------------------------------------------------------------------

    async def _iter_session_keys_async(self) -> list[str]:
        """Enumerate session keys through a public backend API or local index."""
        return await _enumerate_collection_keys(
            self._store,
            collection=SESSIONS_COLLECTION,
            disk_index=self._disk_index,
        )

    # -------------------------------------------------------------------------
    # Async Core Operations
    # -------------------------------------------------------------------------

    async def save_session_async(self, session: ResearchSession) -> None:
        """Save a research session (async)."""
        # Calculate TTL in seconds from now
        ttl: float | None = None
        if session.expires_at is not None:
            ttl = max(1.0, session.expires_at - time.time())

        await self._store.put(
            session.interaction_id,
            session.to_dict(),
            ttl=ttl,
            collection=SESSIONS_COLLECTION,
        )
        if self._disk_index is not None:
            await self._disk_index.add(SESSIONS_COLLECTION, session.interaction_id)
        logger.info(
            "💾 Saved session: %s (expires: %s)",
            session.interaction_id[:16],
            session.time_remaining_human,
        )

    async def get_session_async(self, interaction_id: str) -> ResearchSession | None:
        """Get a session by interaction_id (async)."""
        data = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
        if data is None:
            return None

        session = ResearchSession.from_dict(data)
        # DiskStore handles TTL, but double-check for edge cases
        if session.is_expired:
            logger.debug("Session %s has expired", interaction_id[:16])
            await self._store.delete(interaction_id, collection=SESSIONS_COLLECTION)
            if self._disk_index is not None:
                await self._disk_index.discard_many(
                    SESSIONS_COLLECTION, [interaction_id]
                )
            return None
        return session

    async def list_sessions_async(
        self,
        *,
        include_expired: bool = False,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ResearchSession]:
        """
        List all sessions, optionally filtered (async).

        Args:
            include_expired: Include expired sessions
            tags: Filter by tags (any match)
            limit: Maximum number of sessions to return

        Returns:
            List of sessions, sorted by created_at (newest first)
        """
        sessions: list[ResearchSession] = []

        stale_ids: list[str] = []
        for interaction_id in await self._iter_session_keys_async():
            data = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
            if data is None:
                stale_ids.append(interaction_id)
                continue

            try:
                session = ResearchSession.from_dict(data)
            except KeyError as e:
                logger.warning("Skipping corrupted session %s: %s", interaction_id[:16], e)
                continue

            if not include_expired and session.is_expired:
                await self._store.delete(
                    interaction_id, collection=SESSIONS_COLLECTION
                )
                stale_ids.append(interaction_id)
                continue

            if tags and not any(tag in session.tags for tag in tags):
                continue

            sessions.append(session)

        if self._disk_index is not None:
            await self._disk_index.discard_many(SESSIONS_COLLECTION, stale_ids)

        # Sort by created_at, newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        # Apply limit if positive
        if limit is not None and limit > 0:
            sessions = sessions[:limit]

        return sessions

    async def delete_session_async(self, interaction_id: str) -> bool:
        """Delete a session (async)."""
        exists = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
        if exists is None:
            return False
        await self._store.delete(interaction_id, collection=SESSIONS_COLLECTION)
        if self._disk_index is not None:
            await self._disk_index.discard_many(SESSIONS_COLLECTION, [interaction_id])
        logger.info("🗑️ Deleted session: %s", interaction_id[:16])
        return True

    async def update_session_async(
        self,
        interaction_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        status: ResearchStatus | None = None,
        summary: str | None = None,
        report_text: str | None = None,
        duration_seconds: float | None = None,
        total_tokens: int | None = None,
        plan_text: str | None = None,
        image_export_ids: list[str] | None = None,
    ) -> ResearchSession | None:
        """Update session metadata (async)."""
        session = await self.get_session_async(interaction_id)
        if session is None:
            return None

        if title is not None:
            session.title = title
        if tags is not None:
            session.tags = tags
        if notes is not None:
            session.notes = notes
        if status is not None:
            session.status = status
        if summary is not None:
            session.summary = summary
        if report_text is not None:
            session.report_text = report_text
        if duration_seconds is not None:
            session.duration_seconds = duration_seconds
        if total_tokens is not None:
            session.total_tokens = total_tokens
        if plan_text is not None:
            session.plan_text = plan_text
        if image_export_ids is not None:
            session.image_export_ids = image_export_ids

        await self.save_session_async(session)
        return session

    async def cleanup_expired_async(self) -> int:
        """Remove all expired sessions (async). Returns count of removed sessions."""
        expired_ids: list[str] = []
        stale_ids: list[str] = []

        for interaction_id in await self._iter_session_keys_async():
            data = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
            if data is None:
                stale_ids.append(interaction_id)
                continue
            session = ResearchSession.from_dict(data)
            if session.is_expired:
                expired_ids.append(interaction_id)

        for interaction_id in expired_ids:
            await self._store.delete(interaction_id, collection=SESSIONS_COLLECTION)
        if self._disk_index is not None:
            await self._disk_index.discard_many(
                SESSIONS_COLLECTION, expired_ids + stale_ids
            )

        if expired_ids:
            logger.info("🧹 Cleaned up %d expired sessions", len(expired_ids))

        return len(expired_ids)

    async def search_async(self, query: str, limit: int = 10) -> list[ResearchSession]:
        """Search sessions by query text (searches query and title) (async)."""
        query_lower = query.lower()
        sessions = await self.list_sessions_async()

        matches = []
        for session in sessions:
            in_query = query_lower in session.query.lower()
            in_title = session.title and query_lower in session.title.lower()
            if in_query or in_title:
                matches.append(session)

        return matches[:limit]

    # -------------------------------------------------------------------------
    # Sync Wrappers (for convenience)
    # -------------------------------------------------------------------------

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run async coroutine in sync context."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)
        else:
            # Running inside an async context - schedule and wait
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                result: T = future.result()
                return result

    def save_session(self, session: ResearchSession) -> None:
        """Save a research session (sync wrapper)."""
        self._run_async(self.save_session_async(session))

    def get_session(self, interaction_id: str) -> ResearchSession | None:
        """Get a session by interaction_id (sync wrapper)."""
        return self._run_async(self.get_session_async(interaction_id))

    def list_sessions(
        self,
        *,
        include_expired: bool = False,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ResearchSession]:
        """List all sessions (sync wrapper)."""
        return self._run_async(
            self.list_sessions_async(
                include_expired=include_expired, tags=tags, limit=limit
            )
        )

    def delete_session(self, interaction_id: str) -> bool:
        """Delete a session (sync wrapper)."""
        return self._run_async(self.delete_session_async(interaction_id))

    def update_session(
        self,
        interaction_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        status: ResearchStatus | None = None,
        summary: str | None = None,
        report_text: str | None = None,
        duration_seconds: float | None = None,
        total_tokens: int | None = None,
        plan_text: str | None = None,
        image_export_ids: list[str] | None = None,
    ) -> ResearchSession | None:
        """Update session metadata (sync wrapper)."""
        return self._run_async(
            self.update_session_async(
                interaction_id,
                title=title,
                tags=tags,
                notes=notes,
                status=status,
                summary=summary,
                report_text=report_text,
                duration_seconds=duration_seconds,
                total_tokens=total_tokens,
                plan_text=plan_text,
                image_export_ids=image_export_ids,
            )
        )

    def cleanup_expired(self) -> int:
        """Remove all expired sessions (sync wrapper)."""
        return self._run_async(self.cleanup_expired_async())

    def search(self, query: str, limit: int = 10) -> list[ResearchSession]:
        """Search sessions (sync wrapper)."""
        return self._run_async(self.search_async(query, limit=limit))


# =============================================================================
# Global Storage Instance
# =============================================================================

_storage: SessionStorage | None = None


def get_storage() -> SessionStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = SessionStorage()
    return _storage


# =============================================================================
# Convenience Functions
# =============================================================================


def save_research_session(
    interaction_id: str,
    query: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    report_text: str | None = None,
    format_instructions: str | None = None,
    agent_name: DeepResearchAgent | None = None,
    duration_seconds: float | None = None,
    total_tokens: int | None = None,
    tags: list[str] | None = None,
    status: ResearchStatus = ResearchStatus.COMPLETED,
    plan_text: str | None = None,
    image_export_ids: list[str] | None = None,
) -> ResearchSession:
    """
    Save a research session for later follow-up.

    Args:
        interaction_id: The Gemini interaction ID
        query: The research query
        title: Optional short title (defaults to query[:50])
        summary: Optional AI-generated synopsis for discovery
        report_text: Optional full research report
        format_instructions: Optional format instructions used
        agent_name: Optional agent name used
        duration_seconds: Optional research duration
        total_tokens: Optional total tokens used
        tags: Optional tags for filtering
        status: Session status (default: COMPLETED)
        plan_text: Optional collaborative-planning draft plan text
        image_export_ids: Optional export IDs of persisted Deep Research images

    Returns:
        The saved ResearchSession
    """
    session = ResearchSession(
        interaction_id=interaction_id,
        query=query,
        created_at=time.time(),
        title=title,
        summary=summary,
        report_text=report_text,
        format_instructions=format_instructions,
        agent_name=agent_name,
        duration_seconds=duration_seconds,
        total_tokens=total_tokens,
        tags=tags or [],
        status=status,
        plan_text=plan_text,
        image_export_ids=image_export_ids or [],
    )
    get_storage().save_session(session)
    return session


def update_research_session(
    interaction_id: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    report_text: str | None = None,
    duration_seconds: float | None = None,
    total_tokens: int | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    status: ResearchStatus | None = None,
    plan_text: str | None = None,
    image_export_ids: list[str] | None = None,
) -> ResearchSession | None:
    """
    Update an existing research session.

    Args:
        interaction_id: The Gemini interaction ID
        title: Optional new title
        summary: Optional new summary
        report_text: Optional new report text
        duration_seconds: Optional research duration
        total_tokens: Optional total tokens used
        tags: Optional new tags
        notes: Optional user notes
        status: Optional new status
        plan_text: Optional collaborative-planning draft plan text
        image_export_ids: Optional export IDs of persisted Deep Research images

    Returns:
        The updated ResearchSession or None if not found
    """
    return get_storage().update_session(
        interaction_id,
        title=title,
        summary=summary,
        report_text=report_text,
        duration_seconds=duration_seconds,
        total_tokens=total_tokens,
        tags=tags,
        notes=notes,
        status=status,
        plan_text=plan_text,
        image_export_ids=image_export_ids,
    )


def list_resumable_sessions(limit: int = 10) -> list[ResearchSession]:
    """
    List sessions that can be resumed (in_progress or interrupted).

    Returns:
        List of resumable sessions, sorted by created_at (newest first)
    """
    sessions = get_storage().list_sessions(include_expired=False, limit=None)
    resumable = [s for s in sessions if s.is_resumable]
    return resumable[:limit]


def get_research_session(interaction_id: str) -> ResearchSession | None:
    """Get a research session by interaction_id."""
    return get_storage().get_session(interaction_id)


def delete_research_session(interaction_id: str) -> bool:
    """Delete a research session by interaction_id."""
    return get_storage().delete_session(interaction_id)


def list_research_sessions(
    *,
    include_expired: bool = False,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[ResearchSession]:
    """List research sessions."""
    return get_storage().list_sessions(
        include_expired=include_expired,
        tags=tags,
        limit=limit,
    )


# =============================================================================
# Export Artifact Storage
#
# Backed by the same shared/local backend selection as SessionStorage (see
# create_store()), so an export created on one instance is downloadable from
# another the moment GEMINI_RESEARCH_STORAGE_URL points them at the same
# Redis backend. Replaces the old in-memory `_export_cache` dict, which did
# not survive a restart and could not be shared across workers.
# =============================================================================

# Default TTL for exported files (1 hour) - short-lived, download-once artifacts.
DEFAULT_EXPORT_TTL_SECONDS = 3600


@dataclass
class ExportArtifact:
    """A persisted export artifact (Markdown/JSON/DOCX report bytes + metadata)."""

    export_id: str
    session_id: str
    filename: str
    format: str
    mime_type: str
    content_b64: str  # base64-encoded bytes, for JSON-serializable storage
    created_at: float  # Unix timestamp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportArtifact:
        return cls(
            export_id=data["export_id"],
            session_id=data["session_id"],
            filename=data["filename"],
            format=data["format"],
            mime_type=data["mime_type"],
            content_b64=data["content_b64"],
            created_at=data["created_at"],
        )

    @property
    def content(self) -> bytes:
        import base64

        return base64.b64decode(self.content_b64)

    @property
    def size_human(self) -> str:
        size: float = len(self.content)
        for unit in ["B", "KB", "MB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


class ExportArtifactStore:
    """Persistent, TTL-bounded storage for exported research reports."""

    def __init__(
        self, storage_dir: Path | None = None, *, ttl_seconds: int = DEFAULT_EXPORT_TTL_SECONDS
    ):
        self.storage_dir = storage_dir or get_storage_dir()
        self.ttl_seconds = ttl_seconds
        self._store = create_store(storage_dir=self.storage_dir)
        self._disk_index = (
            _DiskKeyIndex(self.storage_dir) if isinstance(self._store, DiskStore) else None
        )

    async def save_async(
        self,
        *,
        session_id: str,
        filename: str,
        format: str,
        mime_type: str,
        content: bytes,
    ) -> str:
        """Persist an export and return its unique export_id."""
        import base64
        import uuid

        export_id = str(uuid.uuid4())[:12]
        artifact = ExportArtifact(
            export_id=export_id,
            session_id=session_id,
            filename=filename,
            format=format,
            mime_type=mime_type,
            content_b64=base64.b64encode(content).decode("ascii"),
            created_at=time.time(),
        )
        await self._store.put(
            export_id,
            artifact.to_dict(),
            ttl=self.ttl_seconds,
            collection=EXPORTS_COLLECTION,
        )
        if self._disk_index is not None:
            await self._disk_index.add(EXPORTS_COLLECTION, export_id)
        return export_id

    async def get_async(self, export_id: str) -> ExportArtifact | None:
        """Retrieve an export by ID, or None if missing/expired (TTL-enforced by backend)."""
        data = await self._store.get(export_id, collection=EXPORTS_COLLECTION)
        if data is None:
            return None
        return ExportArtifact.from_dict(data)

    async def list_async(self) -> list[ExportArtifact]:
        """List all currently live (non-expired) exports, newest first."""
        export_ids = await _enumerate_collection_keys(
            self._store,
            collection=EXPORTS_COLLECTION,
            disk_index=self._disk_index,
        )

        artifacts: list[ExportArtifact] = []
        stale_ids: list[str] = []
        for export_id in export_ids:
            artifact = await self.get_async(export_id)
            if artifact is not None:
                artifacts.append(artifact)
            else:
                stale_ids.append(export_id)

        if self._disk_index is not None:
            await self._disk_index.discard_many(EXPORTS_COLLECTION, stale_ids)

        artifacts.sort(key=lambda a: a.created_at, reverse=True)
        return artifacts


_export_store: ExportArtifactStore | None = None


def get_export_store() -> ExportArtifactStore:
    """Get the global export artifact store instance."""
    global _export_store
    if _export_store is None:
        _export_store = ExportArtifactStore()
    return _export_store
