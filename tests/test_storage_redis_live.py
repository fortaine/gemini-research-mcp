"""Opt-in live-Redis cross-instance sharing tests.

These tests require a real, reachable Redis/Valkey instance and are skipped
by default (see `redis` marker in pyproject.toml, excluded via `addopts`).

They complement `tests/test_storage.py::TestCrossInstanceSharing`, which
proves the sharing *abstraction* using two instances pointed at the same disk
directory (no live Redis required). These tests prove the same contract
holds against a real `RedisStore` backend, end-to-end.

Run with a local Redis (e.g. `docker run --rm -p 6379:6379 redis:7-alpine`):

    GEMINI_RESEARCH_REDIS_TEST_URL=redis://localhost:6379/0 \
        uv run pytest -m redis tests/test_storage_redis_live.py -q
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from gemini_research_mcp.storage import (
    ExportArtifactStore,
    ResearchSession,
    SessionStorage,
)

REDIS_TEST_URL_ENV_VAR = "GEMINI_RESEARCH_REDIS_TEST_URL"

pytestmark = pytest.mark.redis


@pytest.fixture(autouse=True)
def _redis_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point GEMINI_RESEARCH_STORAGE_URL at the live test Redis, or skip."""
    redis_url = os.environ.get(REDIS_TEST_URL_ENV_VAR)
    if not redis_url:
        pytest.skip(
            f"{REDIS_TEST_URL_ENV_VAR} not set - live Redis sharing tests are opt-in. "
            "Start a local Redis (e.g. `docker run --rm -p 6379:6379 redis:7-alpine`) "
            f"and set {REDIS_TEST_URL_ENV_VAR}=redis://localhost:6379/0 to run them."
        )
    monkeypatch.setenv("GEMINI_RESEARCH_STORAGE_URL", redis_url)


class TestLiveRedisSessionSharing:
    """A session saved by one SessionStorage instance is readable by another
    over a real Redis backend - the distributed-deployment acceptance
    criterion from plan.md chantier 4, verified against live infrastructure
    rather than only the backend-selection abstraction."""

    @pytest.mark.asyncio
    async def test_session_created_by_a_is_readable_by_b_over_redis(self) -> None:
        from key_value.aio.stores.redis.store import RedisStore

        storage_a = SessionStorage()
        assert isinstance(storage_a._store, RedisStore)  # noqa: SLF001 - verifying real backend wiring

        interaction_id = f"live-redis-{uuid.uuid4().hex}"
        session = ResearchSession(
            interaction_id=interaction_id,
            query="live redis sharing smoke test",
            created_at=time.time(),
        )
        await storage_a.save_session_async(session)

        storage_b = SessionStorage()
        retrieved = await storage_b.get_session_async(interaction_id)

        assert retrieved is not None
        assert retrieved.interaction_id == interaction_id
        assert retrieved.query == "live redis sharing smoke test"

    @pytest.mark.asyncio
    async def test_session_list_visible_across_instances_over_redis(self) -> None:
        storage_a = SessionStorage()
        ids = [f"live-redis-list-{uuid.uuid4().hex}" for _ in range(3)]
        for interaction_id in ids:
            await storage_a.save_session_async(
                ResearchSession(
                    interaction_id=interaction_id,
                    query="live redis list smoke test",
                    created_at=time.time(),
                )
            )

        storage_b = SessionStorage()
        sessions = await storage_b.list_sessions_async()
        listed_ids = {s.interaction_id for s in sessions}
        assert set(ids).issubset(listed_ids)


class TestLiveRedisExportSharing:
    """An export artifact saved by one ExportArtifactStore instance is
    downloadable from another over a real Redis backend."""

    @pytest.mark.asyncio
    async def test_export_created_by_a_is_readable_by_b_over_redis(self) -> None:
        store_a = ExportArtifactStore()
        export_id = await store_a.save_async(
            session_id="live-redis-session",
            filename="report.md",
            format="markdown",
            mime_type="text/markdown",
            content=b"hello redis",
        )

        store_b = ExportArtifactStore()
        retrieved = await store_b.get_async(export_id)

        assert retrieved is not None
        assert retrieved.export_id == export_id
        assert retrieved.content == b"hello redis"
