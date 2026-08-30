"""Tests for resume_research session cleanup behavior.

Tests the auto-deletion of stale and not-found sessions when resuming.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gemini_research_mcp.storage import (
    ResearchSession,
    ResearchStatus,
    SessionStorage,
    delete_research_session,
    get_research_session,
    save_research_session,
)

# =============================================================================
# Fixtures
# =============================================================================

STORAGE_MODULE = "gemini_research_mcp.storage"
SERVER_MODULE = "gemini_research_mcp.server"


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a temporary storage directory for all tests."""
    storage = SessionStorage(storage_dir=tmp_path)
    monkeypatch.setattr(f"{STORAGE_MODULE}._storage", storage)


@pytest.fixture
def in_progress_session() -> ResearchSession:
    """Create an in-progress session (recent, <1h old)."""
    return ResearchSession(
        interaction_id="test-recent-session",
        query="Recent research query",
        created_at=time.time() - 1800,  # 30 min ago
        status=ResearchStatus.IN_PROGRESS,
    )


@pytest.fixture
def stale_session() -> ResearchSession:
    """Create a stale in-progress session (>24h old)."""
    return ResearchSession(
        interaction_id="test-stale-session",
        query="Very old stuck research query",
        created_at=time.time() - 90000,  # ~25h ago
        status=ResearchStatus.IN_PROGRESS,
    )


# =============================================================================
# delete_research_session convenience function
# =============================================================================


class TestDeleteResearchSession:
    """Tests for the delete_research_session convenience function."""

    def test_delete_existing_session(self) -> None:
        """Should delete an existing session and return True."""
        save_research_session(
            interaction_id="to-delete",
            query="delete me",
        )
        assert get_research_session("to-delete") is not None

        result = delete_research_session("to-delete")
        assert result is True
        assert get_research_session("to-delete") is None

    def test_delete_nonexistent_session(self) -> None:
        """Should return False for nonexistent session."""
        result = delete_research_session("does-not-exist")
        assert result is False


# =============================================================================
# resume_research — stale session cleanup (>24h)
# =============================================================================


class TestResumeResearchStaleCleanup:
    """Tests for auto-deletion of stale sessions stuck in_progress >24h."""

    @pytest.mark.asyncio
    async def test_stale_session_deleted(
        self, stale_session: ResearchSession
    ) -> None:
        """Sessions stuck in_progress >24h should be auto-deleted."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id=stale_session.interaction_id,
            query=stale_session.query,
            status=ResearchStatus.IN_PROGRESS,
        )
        # Manually set created_at to make it stale
        from gemini_research_mcp.storage import get_storage

        storage = get_storage()
        session = storage.get_session(stale_session.interaction_id)
        assert session is not None
        session.created_at = stale_session.created_at
        storage.save_session(session)

        # Mock Gemini API to return "still in_progress"
        mock_interaction = SimpleNamespace(status="in_progress")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text=None,
            usage=None,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await resume_research(interaction_id=stale_session.interaction_id)

        data = json.loads(result)
        assert data["status"] == "deleted_stale"
        assert "stuck in progress" in data["message"]
        assert get_research_session(stale_session.interaction_id) is None

    @pytest.mark.asyncio
    async def test_recent_session_not_deleted(
        self, in_progress_session: ResearchSession
    ) -> None:
        """Sessions in_progress <24h should NOT be deleted."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id=in_progress_session.interaction_id,
            query=in_progress_session.query,
            status=ResearchStatus.IN_PROGRESS,
        )

        # Mock Gemini API to return "still in_progress"
        mock_interaction = SimpleNamespace(status="in_progress")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text=None,
            usage=None,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await resume_research(
                interaction_id=in_progress_session.interaction_id
            )

        data = json.loads(result)
        assert data["status"] == "still_in_progress"
        # Session should still exist
        assert get_research_session(in_progress_session.interaction_id) is not None

    @pytest.mark.asyncio
    async def test_no_arg_listing_separates_recent_failed_and_stale_sessions(self) -> None:
        """Listing should surface recent failed sessions without deleting stale ones."""
        from gemini_research_mcp.server import resume_research
        from gemini_research_mcp.storage import get_storage

        save_research_session(
            interaction_id="recent-failed-session",
            query="Recent failed gateway timeout",
            status=ResearchStatus.FAILED,
        )
        save_research_session(
            interaction_id="stale-in-progress-session",
            query="Old in-progress session",
            status=ResearchStatus.IN_PROGRESS,
        )

        storage = get_storage()
        stale = storage.get_session("stale-in-progress-session")
        assert stale is not None
        stale.created_at = time.time() - 90000
        storage.save_session(stale)

        result = await resume_research()
        data = json.loads(result)

        assert data["status"] == "recent_failed_sessions_found"
        assert data["recent_failed_sessions"][0]["interaction_id"] == "recent-failed-session"
        assert data["stale_sessions"][0]["interaction_id"] == "stale-in-progress-session"
        assert get_research_session("stale-in-progress-session") is not None


# =============================================================================
# resume_research — Gemini API not-found cleanup
# =============================================================================


class TestResumeResearchNotFoundCleanup:
    """Tests for auto-deletion when Gemini API returns 404/not_found."""

    @pytest.mark.asyncio
    async def test_404_deletes_session(self) -> None:
        """Session should be deleted when Gemini returns 404."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="test-404-session",
            query="Query that Gemini forgot",
            status=ResearchStatus.IN_PROGRESS,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            side_effect=Exception("404 NOT_FOUND: interaction not found"),
        ):
            result = await resume_research(interaction_id="test-404-session")

        data = json.loads(result)
        assert data["status"] == "deleted_not_found"
        assert "no longer exists" in data["message"]
        assert get_research_session("test-404-session") is None

    @pytest.mark.asyncio
    async def test_not_found_error_deletes_session(self) -> None:
        """Session should be deleted when error contains 'not_found'."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="test-notfound-session",
            query="Another lost query",
            status=ResearchStatus.IN_PROGRESS,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            side_effect=Exception("not_found: resource does not exist"),
        ):
            result = await resume_research(interaction_id="test-notfound-session")

        data = json.loads(result)
        assert data["status"] == "deleted_not_found"
        assert get_research_session("test-notfound-session") is None

    @pytest.mark.asyncio
    async def test_generic_error_does_not_delete(self) -> None:
        """Generic API errors should NOT delete the session."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="test-generic-error",
            query="Temporary network issue",
            status=ResearchStatus.IN_PROGRESS,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            side_effect=Exception("Connection timed out"),
        ):
            result = await resume_research(interaction_id="test-generic-error")

        data = json.loads(result)
        assert data["status"] == "api_error"
        # Session should still exist (marked interrupted)
        session = get_research_session("test-generic-error")
        assert session is not None
        assert session.status == ResearchStatus.INTERRUPTED

    @pytest.mark.asyncio
    async def test_completed_without_report_stays_visible_for_recovery(self) -> None:
        """A completed Gemini status with empty text should not be marked completed."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="empty-completed-session",
            query="Query that completed without report text",
            status=ResearchStatus.IN_PROGRESS,
        )

        mock_interaction = SimpleNamespace(status="completed")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text="",
            usage=SimpleNamespace(total_tokens=123),
        )

        async def passthrough_citations(result: object, resolve_urls: bool) -> object:
            del resolve_urls
            return result

        with (
            patch(
                f"{SERVER_MODULE}.get_research_status",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                f"{SERVER_MODULE}.process_citations",
                new=passthrough_citations,
            ),
        ):
            result = await resume_research(interaction_id="empty-completed-session")

        data = json.loads(result)
        assert data["status"] == "completed_without_report"
        assert data["resumable"] is True

        session = get_research_session("empty-completed-session")
        assert session is not None
        assert session.status == ResearchStatus.INTERRUPTED
        assert session.report_text is None
        assert session.total_tokens == 123

    @pytest.mark.asyncio
    async def test_already_completed_without_stored_report_rechecks_gemini(self) -> None:
        """Completed local sessions without report text should not short-circuit resume."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="completed-empty-session",
            query="Completed locally but report was not stored",
            report_text="",
            status=ResearchStatus.COMPLETED,
        )

        mock_interaction = SimpleNamespace(status="completed")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text="",
            usage=None,
        )

        async def passthrough_citations(result: object, resolve_urls: bool) -> object:
            del resolve_urls
            return result

        with (
            patch(
                f"{SERVER_MODULE}.get_research_status",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as get_status,
            patch(
                f"{SERVER_MODULE}.process_citations",
                new=passthrough_citations,
            ),
        ):
            result = await resume_research(interaction_id="completed-empty-session")

        data = json.loads(result)
        assert data["status"] == "completed_without_report"
        get_status.assert_awaited_once_with("completed-empty-session")

        session = get_research_session("completed-empty-session")
        assert session is not None
        assert session.status == ResearchStatus.INTERRUPTED


# =============================================================================
# Startup log — quiet mode
# =============================================================================


class TestStartupLogging:
    """Tests for the quieter startup log."""

    def test_startup_checks_resumable_count(self) -> None:
        """Startup should log a single-line resumable count, not per-session."""
        # Save a few in-progress sessions
        for i in range(3):
            save_research_session(
                interaction_id=f"startup-test-{i}",
                query=f"Query {i}",
                status=ResearchStatus.IN_PROGRESS,
            )

        from gemini_research_mcp.storage import list_resumable_sessions

        resumable = list_resumable_sessions(limit=10)
        assert len(resumable) == 3
