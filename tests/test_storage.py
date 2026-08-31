"""Tests for session storage using py-key-value-aio DiskStore."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from gemini_research_mcp.storage import (
    DEFAULT_TTL_SECONDS,
    DISK_INDEX_FILENAME,
    SESSIONS_COLLECTION,
    ExportArtifactStore,
    ResearchSession,
    SessionStorage,
    get_storage_dir,
    get_ttl_seconds,
)
from gemini_research_mcp.types import DeepResearchAgent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage(tmp_path: Path) -> SessionStorage:
    """Create a temporary storage instance for testing."""
    return SessionStorage(storage_dir=tmp_path)


@pytest.fixture
def sample_session() -> ResearchSession:
    """Create a sample research session."""
    return ResearchSession(
        interaction_id="test-interaction-12345",
        query="What is quantum computing?",
        created_at=time.time(),
        title="Quantum Computing Research",
        agent_name=DeepResearchAgent.DEEP_RESEARCH_PRO,
        duration_seconds=45.5,
        total_tokens=1500,
        tags=["physics", "computing"],
        notes="Initial research on quantum mechanics",
    )


# =============================================================================
# ResearchSession Tests
# =============================================================================


class TestResearchSession:
    """Tests for the ResearchSession dataclass."""

    def test_session_creation(self, sample_session: ResearchSession) -> None:
        """Test basic session creation."""
        assert sample_session.interaction_id == "test-interaction-12345"
        assert sample_session.query == "What is quantum computing?"
        assert sample_session.title == "Quantum Computing Research"
        assert sample_session.tags == ["physics", "computing"]

    def test_expires_at_auto_set(self) -> None:
        """Test that expires_at is automatically set based on TTL."""
        now = time.time()
        session = ResearchSession(
            interaction_id="test-123",
            query="Test query",
            created_at=now,
        )
        assert session.expires_at is not None
        assert session.expires_at == pytest.approx(now + get_ttl_seconds(), abs=1)

    def test_is_expired_false(self, sample_session: ResearchSession) -> None:
        """Test that fresh session is not expired."""
        assert sample_session.is_expired is False

    def test_is_expired_true(self) -> None:
        """Test that old session is expired."""
        past_time = time.time() - 100  # 100 seconds ago
        session = ResearchSession(
            interaction_id="test-123",
            query="Test query",
            created_at=past_time,
            expires_at=past_time + 10,  # Expired 90 seconds ago
        )
        assert session.is_expired is True

    def test_time_remaining(self, sample_session: ResearchSession) -> None:
        """Test time remaining calculation."""
        remaining = sample_session.time_remaining
        assert remaining is not None
        assert remaining > 0
        assert remaining <= DEFAULT_TTL_SECONDS

    def test_time_remaining_human(self, sample_session: ResearchSession) -> None:
        """Test human-readable time remaining."""
        human = sample_session.time_remaining_human
        assert human is not None
        assert "d" in human  # Should show days for 55-day TTL

    def test_time_remaining_expired(self) -> None:
        """Test time remaining for expired session."""
        session = ResearchSession(
            interaction_id="test-123",
            query="Test query",
            created_at=time.time() - 100,
            expires_at=time.time() - 10,
        )
        assert session.time_remaining_human == "expired"

    def test_to_dict(self, sample_session: ResearchSession) -> None:
        """Test serialization to dict."""
        data = sample_session.to_dict()
        assert data["interaction_id"] == "test-interaction-12345"
        assert data["query"] == "What is quantum computing?"
        assert data["tags"] == ["physics", "computing"]

    def test_from_dict(self, sample_session: ResearchSession) -> None:
        """Test deserialization from dict."""
        data = sample_session.to_dict()
        restored = ResearchSession.from_dict(data)
        assert restored.interaction_id == sample_session.interaction_id
        assert restored.query == sample_session.query
        assert restored.tags == sample_session.tags

    def test_short_description(self, sample_session: ResearchSession) -> None:
        """Test short description generation."""
        desc = sample_session.short_description()
        assert "test-intera" in desc
        assert "Quantum Computing Research" in desc


# =============================================================================
# SessionStorage Async Tests
# =============================================================================


class TestSessionStorageAsync:
    """Tests for async SessionStorage operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_session(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        """Test saving and retrieving a session."""
        await temp_storage.save_session_async(sample_session)
        retrieved = await temp_storage.get_session_async(sample_session.interaction_id)

        assert retrieved is not None
        assert retrieved.interaction_id == sample_session.interaction_id
        assert retrieved.query == sample_session.query
        assert retrieved.title == sample_session.title

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, temp_storage: SessionStorage) -> None:
        """Test getting a session that doesn't exist."""
        result = await temp_storage.get_session_async("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(3):
            session = ResearchSession(
                interaction_id=f"test-{i}",
                query=f"Query {i}",
                created_at=time.time() - i * 100,  # Different timestamps
            )
            await temp_storage.save_session_async(session)

        sessions = await temp_storage.list_sessions_async()
        assert len(sessions) == 3
        # Should be sorted by created_at, newest first
        assert sessions[0].interaction_id == "test-0"

    @pytest.mark.asyncio
    async def test_save_creates_versioned_disk_index(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        await temp_storage.save_session_async(sample_session)

        index_path = temp_storage.storage_dir / DISK_INDEX_FILENAME
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert index["version"] == 1
        assert sample_session.interaction_id in index["collections"]["sessions"]

    @pytest.mark.asyncio
    async def test_listing_migrates_legacy_disk_cache(self, tmp_path: Path) -> None:
        from key_value.aio.stores.disk import DiskStore

        legacy_session = ResearchSession(
            interaction_id="legacy-v0160b1",
            query="Legacy session",
            created_at=time.time(),
        )
        legacy_store = DiskStore(directory=tmp_path)
        await legacy_store.put(
            legacy_session.interaction_id,
            legacy_session.to_dict(),
            collection=SESSIONS_COLLECTION,
        )

        storage = SessionStorage(storage_dir=tmp_path)
        sessions = await storage.list_sessions_async()

        assert [session.interaction_id for session in sessions] == [
            legacy_session.interaction_id
        ]
        index = json.loads(
            (tmp_path / DISK_INDEX_FILENAME).read_text(encoding="utf-8")
        )
        assert legacy_session.interaction_id in index["collections"]["sessions"]
        await legacy_store.close()

    @pytest.mark.asyncio
    async def test_listing_rebuilds_corrupted_disk_index(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        await temp_storage.save_session_async(sample_session)
        index_path = temp_storage.storage_dir / DISK_INDEX_FILENAME
        index_path.write_text("{not-json", encoding="utf-8")

        storage = SessionStorage(storage_dir=temp_storage.storage_dir)
        sessions = await storage.list_sessions_async()

        assert [session.interaction_id for session in sessions] == [
            sample_session.interaction_id
        ]
        rebuilt = json.loads(index_path.read_text(encoding="utf-8"))
        assert rebuilt["version"] == 1

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test listing sessions with limit."""
        for i in range(5):
            session = ResearchSession(
                interaction_id=f"test-{i}",
                query=f"Query {i}",
                created_at=time.time() - i * 100,
            )
            await temp_storage.save_session_async(session)

        sessions = await temp_storage.list_sessions_async(limit=2)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_with_tags(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test filtering sessions by tags."""
        session1 = ResearchSession(
            interaction_id="test-1",
            query="Query 1",
            created_at=time.time(),
            tags=["ai", "research"],
        )
        session2 = ResearchSession(
            interaction_id="test-2",
            query="Query 2",
            created_at=time.time(),
            tags=["physics"],
        )
        await temp_storage.save_session_async(session1)
        await temp_storage.save_session_async(session2)

        sessions = await temp_storage.list_sessions_async(tags=["ai"])
        assert len(sessions) == 1
        assert sessions[0].interaction_id == "test-1"

    @pytest.mark.asyncio
    async def test_list_sessions_skips_corrupted_entries(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        """Test that listing sessions skips corrupted entries gracefully."""
        # Save a valid session
        await temp_storage.save_session_async(sample_session)
        await temp_storage.save_session_async(
            ResearchSession(
                interaction_id="corrupted-123",
                query="Will be corrupted",
                created_at=time.time(),
            )
        )

        # Manually write a corrupted entry (missing required 'query' field)
        corrupted_data = {"interaction_id": "corrupted-123", "created_at": 12345.0}
        await temp_storage._store.put(
            "corrupted-123",
            corrupted_data,
            collection="sessions",
        )

        # Listing should succeed and only return the valid session
        sessions = await temp_storage.list_sessions_async()
        assert len(sessions) == 1
        assert sessions[0].interaction_id == sample_session.interaction_id

    @pytest.mark.asyncio
    async def test_delete_session(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        """Test deleting a session."""
        await temp_storage.save_session_async(sample_session)
        
        # Verify it exists
        assert await temp_storage.get_session_async(sample_session.interaction_id) is not None
        
        # Delete it
        result = await temp_storage.delete_session_async(sample_session.interaction_id)
        assert result is True
        
        # Verify it's gone
        assert await temp_storage.get_session_async(sample_session.interaction_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test deleting a session that doesn't exist."""
        result = await temp_storage.delete_session_async("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_session(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        """Test updating session metadata."""
        await temp_storage.save_session_async(sample_session)

        updated = await temp_storage.update_session_async(
            sample_session.interaction_id,
            title="Updated Title",
            tags=["new-tag"],
            notes="Updated notes",
        )

        assert updated is not None
        assert updated.title == "Updated Title"
        assert updated.tags == ["new-tag"]
        assert updated.notes == "Updated notes"

        # Verify persistence
        retrieved = await temp_storage.get_session_async(sample_session.interaction_id)
        assert retrieved is not None
        assert retrieved.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_search_sessions(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test searching sessions."""
        session1 = ResearchSession(
            interaction_id="test-1",
            query="Quantum computing basics",
            created_at=time.time(),
            title="Quantum Research",
        )
        session2 = ResearchSession(
            interaction_id="test-2",
            query="Machine learning models",
            created_at=time.time(),
        )
        await temp_storage.save_session_async(session1)
        await temp_storage.save_session_async(session2)

        # Search by query
        results = await temp_storage.search_async("quantum")
        assert len(results) == 1
        assert results[0].interaction_id == "test-1"

        # Search by title
        results = await temp_storage.search_async("research")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test cleanup of expired sessions."""
        # Create an expired session
        expired = ResearchSession(
            interaction_id="expired-1",
            query="Old query",
            created_at=time.time() - 100,
            expires_at=time.time() - 10,  # Already expired
        )
        # Create a valid session
        valid = ResearchSession(
            interaction_id="valid-1",
            query="New query",
            created_at=time.time(),
        )

        await temp_storage.save_session_async(expired)
        await temp_storage.save_session_async(valid)

        # Cleanup
        count = await temp_storage.cleanup_expired_async()
        assert count == 1

        # Verify expired is gone, valid remains
        assert await temp_storage.get_session_async("expired-1") is None
        assert await temp_storage.get_session_async("valid-1") is not None


# =============================================================================
# SessionStorage Sync Wrapper Tests
# =============================================================================


class TestSessionStorageSync:
    """Tests for sync wrapper methods."""

    def test_save_and_get_session_sync(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        """Test sync save and get."""
        temp_storage.save_session(sample_session)
        retrieved = temp_storage.get_session(sample_session.interaction_id)

        assert retrieved is not None
        assert retrieved.interaction_id == sample_session.interaction_id

    def test_list_sessions_sync(
        self, temp_storage: SessionStorage
    ) -> None:
        """Test sync list sessions."""
        session = ResearchSession(
            interaction_id="test-1",
            query="Query 1",
            created_at=time.time(),
        )
        temp_storage.save_session(session)

        sessions = temp_storage.list_sessions()
        assert len(sessions) == 1

    def test_delete_session_sync(
        self, temp_storage: SessionStorage, sample_session: ResearchSession
    ) -> None:
        """Test sync delete."""
        temp_storage.save_session(sample_session)
        result = temp_storage.delete_session(sample_session.interaction_id)
        assert result is True


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for configuration functions."""

    def test_get_storage_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default storage directory."""
        monkeypatch.delenv("GEMINI_RESEARCH_STORAGE_PATH", raising=False)
        storage_dir = get_storage_dir()
        assert "gemini-research-mcp" in str(storage_dir)

    def test_get_storage_dir_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test custom storage path from environment."""
        monkeypatch.setenv("GEMINI_RESEARCH_STORAGE_PATH", "/custom/path/sessions.json")
        storage_dir = get_storage_dir()
        assert storage_dir == Path("/custom/path")

    def test_get_ttl_seconds_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default TTL."""
        monkeypatch.delenv("GEMINI_RESEARCH_TTL_SECONDS", raising=False)
        ttl = get_ttl_seconds()
        assert ttl == DEFAULT_TTL_SECONDS

    def test_get_ttl_seconds_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test custom TTL from environment."""
        monkeypatch.setenv("GEMINI_RESEARCH_TTL_SECONDS", "3600")
        ttl = get_ttl_seconds()
        assert ttl == 3600


# =============================================================================
# Storage Backend Selection Tests (GEMINI_RESEARCH_STORAGE_URL)
# =============================================================================


class TestBackendSelection:
    """Tests for create_store()'s local-disk vs shared-Redis backend selection."""

    def test_no_url_selects_disk_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without GEMINI_RESEARCH_STORAGE_URL, the local disk backend is used."""
        from key_value.aio.stores.disk import DiskStore

        from gemini_research_mcp.storage import create_store

        monkeypatch.delenv("GEMINI_RESEARCH_STORAGE_URL", raising=False)
        store = create_store(storage_dir=tmp_path)
        assert isinstance(store, DiskStore)

    def test_url_selects_redis_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting GEMINI_RESEARCH_STORAGE_URL switches to the shared Redis backend.

        RedisStore's constructor is lazy (no eager connection attempt), so this
        can be verified without a running Redis server.
        """
        from key_value.aio.stores.redis.store import RedisStore

        from gemini_research_mcp.storage import create_store

        monkeypatch.setenv("GEMINI_RESEARCH_STORAGE_URL", "redis://localhost:6379/0")
        store = create_store()
        assert isinstance(store, RedisStore)

    def test_get_storage_backend_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_storage_backend_url reflects the env var, None when unset/empty."""
        from gemini_research_mcp.storage import get_storage_backend_url

        monkeypatch.delenv("GEMINI_RESEARCH_STORAGE_URL", raising=False)
        assert get_storage_backend_url() is None

        monkeypatch.setenv("GEMINI_RESEARCH_STORAGE_URL", "redis://localhost:6379/0")
        assert get_storage_backend_url() == "redis://localhost:6379/0"

        monkeypatch.setenv("GEMINI_RESEARCH_STORAGE_URL", "")
        assert get_storage_backend_url() is None


# =============================================================================
# Cross-Instance Sharing Tests (multi-worker / stateless deployment proof)
# =============================================================================


class TestCrossInstanceSharing:
    """Prove a session created by one SessionStorage instance is readable by another.

    This is the acceptance criterion from the stateless-storage modernization
    plan: "A research session started on one instance is resumable from
    another with a shared backend." Using two instances pointed at the same
    disk directory exercises the same code path a Redis-shared deployment
    would use (the abstraction doesn't distinguish instances, only backends).
    """

    @pytest.mark.asyncio
    async def test_session_created_by_a_is_readable_by_b(
        self, tmp_path: Path, sample_session: ResearchSession
    ) -> None:
        storage_a = SessionStorage(storage_dir=tmp_path)
        await storage_a.save_session_async(sample_session)

        storage_b = SessionStorage(storage_dir=tmp_path)
        retrieved = await storage_b.get_session_async(sample_session.interaction_id)

        assert retrieved is not None
        assert retrieved.interaction_id == sample_session.interaction_id

    @pytest.mark.asyncio
    async def test_session_list_visible_across_instances(self, tmp_path: Path) -> None:
        storage_a = SessionStorage(storage_dir=tmp_path)
        for i in range(3):
            await storage_a.save_session_async(
                ResearchSession(
                    interaction_id=f"shared-{i}",
                    query=f"Query {i}",
                    created_at=time.time(),
                )
            )

        storage_b = SessionStorage(storage_dir=tmp_path)
        sessions = await storage_b.list_sessions_async()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_update_by_b_is_visible_to_a(
        self, tmp_path: Path, sample_session: ResearchSession
    ) -> None:
        storage_a = SessionStorage(storage_dir=tmp_path)
        await storage_a.save_session_async(sample_session)

        storage_b = SessionStorage(storage_dir=tmp_path)
        await storage_b.update_session_async(sample_session.interaction_id, notes="from B")

        refreshed = await storage_a.get_session_async(sample_session.interaction_id)
        assert refreshed is not None
        assert refreshed.notes == "from B"


class TestExportArtifactIndex:
    """Verify local export enumeration uses the project-owned disk index."""

    @pytest.mark.asyncio
    async def test_export_listing_survives_store_recreation(self, tmp_path: Path) -> None:
        store_a = ExportArtifactStore(storage_dir=tmp_path)
        export_id = await store_a.save_async(
            session_id="session-1",
            filename="report.md",
            format="markdown",
            mime_type="text/markdown",
            content=b"report",
        )

        store_b = ExportArtifactStore(storage_dir=tmp_path)
        exports = await store_b.list_async()

        assert [artifact.export_id for artifact in exports] == [export_id]
        index = json.loads(
            (tmp_path / DISK_INDEX_FILENAME).read_text(encoding="utf-8")
        )
        assert export_id in index["collections"]["exports"]

    @pytest.mark.asyncio
    async def test_concurrent_session_and_export_updates_share_index(
        self, tmp_path: Path
    ) -> None:
        session_storage = SessionStorage(storage_dir=tmp_path)
        export_storage = ExportArtifactStore(storage_dir=tmp_path)

        async def save_session(index: int) -> None:
            await session_storage.save_session_async(
                ResearchSession(
                    interaction_id=f"session-{index}",
                    query=f"Query {index}",
                    created_at=time.time(),
                )
            )

        async def save_export(index: int) -> None:
            await export_storage.save_async(
                session_id=f"session-{index}",
                filename=f"report-{index}.md",
                format="markdown",
                mime_type="text/markdown",
                content=f"report {index}".encode(),
            )

        await asyncio.gather(
            *(save_session(index) for index in range(10)),
            *(save_export(index) for index in range(10)),
        )

        index = json.loads(
            (tmp_path / DISK_INDEX_FILENAME).read_text(encoding="utf-8")
        )
        assert len(index["collections"]["sessions"]) == 10
        assert len(index["collections"]["exports"]) == 10
