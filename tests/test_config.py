"""Unit tests for configuration module.

Tests environment variable handling, defaults, and helper functions.
Run with: uv run pytest tests/test_config.py -v
"""

import os
from datetime import date
from unittest.mock import patch

import pytest

from gemini_research_mcp.config import (
    DEFAULT_DEEP_RESEARCH_AGENT,
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    LOGGER_NAME,
    RETRYABLE_ERRORS,
    default_system_prompt,
    get_api_key,
    get_deep_research_agent,
    get_model,
    get_tasks_backend_url,
    is_retryable_error,
)
from gemini_research_mcp.types import DeepResearchAgent


class TestConstants:
    """Test module-level constants."""

    def test_default_model(self):
        """Default model should be gemini-3.8-flash."""
        assert DEFAULT_MODEL == "gemini-3.8-flash"

    def test_default_agent(self):
        """Default agent should be deep-research-pro-preview."""
        assert "deep-research" in DEFAULT_DEEP_RESEARCH_AGENT

    def test_default_thinking_level(self):
        """Default thinking level should be 'high'."""
        assert DEFAULT_THINKING_LEVEL == "high"

    def test_logger_name(self):
        """Logger name should be gemini-research-mcp."""
        assert LOGGER_NAME == "gemini-research-mcp"

    def test_retryable_errors_list(self):
        """Should have expected retryable errors."""
        assert "timeout" in RETRYABLE_ERRORS
        assert "gateway_timeout" in RETRYABLE_ERRORS
        assert "connection_reset" in RETRYABLE_ERRORS


class TestGetApiKey:
    """Test get_api_key function."""

    def test_returns_key_when_set(self):
        """Should return API key from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            assert get_api_key() == "test-key-123"

    def test_raises_when_not_set(self):
        """Should raise ValueError when API key not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                get_api_key()

    def test_raises_for_empty_key(self):
        """Should raise ValueError for empty API key."""
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
            pytest.raises(ValueError, match="GEMINI_API_KEY"),
        ):
            get_api_key()


class TestGetModel:
    """Test get_model function."""

    def test_returns_default_when_not_set(self):
        """Should return default model when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_MODEL", None)
            assert get_model() == DEFAULT_MODEL

    def test_returns_env_override(self):
        """Should return env override when set."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-custom-model"}):
            assert get_model() == "gemini-custom-model"


class TestGetTasksBackendUrl:
    """Test local and distributed Tasks backend resolution."""

    def test_local_default_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_tasks_backend_url() is None

    def test_reuses_shared_redis_url(self):
        with patch.dict(
            os.environ,
            {"GEMINI_RESEARCH_STORAGE_URL": "redis://cache:6379/0"},
            clear=True,
        ):
            assert get_tasks_backend_url() == "redis://cache:6379/0"

    def test_explicit_docket_url_wins(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_RESEARCH_STORAGE_URL": "redis://shared:6379/0",
                "FASTMCP_DOCKET_URL": "redis://tasks:6379/1",
            },
            clear=True,
        ):
            assert get_tasks_backend_url() == "redis://tasks:6379/1"

    def test_non_redis_storage_url_is_not_reused(self):
        with patch.dict(
            os.environ,
            {"GEMINI_RESEARCH_STORAGE_URL": "file:///tmp/research"},
            clear=True,
        ):
            assert get_tasks_backend_url() is None


class TestGetDeepResearchAgent:
    """Test get_deep_research_agent function."""

    def test_returns_default_when_not_set(self):
        """Should return default agent when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEEP_RESEARCH_AGENT", None)
            assert get_deep_research_agent() == DEFAULT_DEEP_RESEARCH_AGENT

    def test_returns_enum_type(self):
        """Should return DeepResearchAgent enum type."""
        agent = get_deep_research_agent()
        assert isinstance(agent, DeepResearchAgent)
        assert agent == DeepResearchAgent.DEEP_RESEARCH

    def test_env_override_aliases(self):
        """Environment variable override should select supported agents."""
        with patch.dict(os.environ, {"DEEP_RESEARCH_AGENT": "max"}):
            assert get_deep_research_agent() == DeepResearchAgent.DEEP_RESEARCH_MAX
        with patch.dict(os.environ, {"DEEP_RESEARCH_AGENT": "pro"}):
            assert get_deep_research_agent() == DeepResearchAgent.DEEP_RESEARCH_PRO

    def test_env_override_rejects_unknown_agent(self):
        """Unknown Deep Research agent values should fail fast."""
        with (
            patch.dict(os.environ, {"DEEP_RESEARCH_AGENT": "custom-agent"}),
            pytest.raises(ValueError, match="Invalid DEEP_RESEARCH_AGENT"),
        ):
            get_deep_research_agent()


class TestIsRetryableError:
    """Test is_retryable_error function."""

    def test_retryable_errors_detected(self):
        """Known retryable errors should be detected."""
        assert is_retryable_error("gateway_timeout: server unavailable")
        assert is_retryable_error("Request timeout after 30s")
        assert is_retryable_error("connection_reset by peer")
        assert is_retryable_error("stream closed unexpectedly")
        assert is_retryable_error("service_unavailable: try again")

    def test_non_retryable_errors(self):
        """Non-retryable errors should not be detected."""
        assert not is_retryable_error("invalid_api_key")
        assert not is_retryable_error("quota_exceeded")
        assert not is_retryable_error("permission_denied")
        assert not is_retryable_error("bad_request: invalid parameter")

    def test_case_insensitive(self):
        """Detection should be case insensitive."""
        assert is_retryable_error("TIMEOUT")
        assert is_retryable_error("Gateway_Timeout")
        assert is_retryable_error("CONNECTION_RESET")


class TestDefaultSystemPrompt:
    """Test default_system_prompt function."""

    def test_includes_current_date(self):
        """Prompt should include current date."""
        prompt = default_system_prompt()
        today = date.today().strftime("%B %d, %Y")
        assert today in prompt

    def test_establishes_role(self):
        """Prompt should establish research analyst role."""
        prompt = default_system_prompt()
        assert "expert" in prompt.lower()
        assert "research" in prompt.lower()

    def test_mentions_citations(self):
        """Prompt should mention citing sources."""
        prompt = default_system_prompt()
        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_reasonable_length(self):
        """Prompt should be reasonably sized."""
        prompt = default_system_prompt()
        # Should be substantial but not too long
        assert 200 < len(prompt) < 2000

    def test_structure_guidance(self):
        """Prompt should mention structuring answers."""
        prompt = default_system_prompt()
        assert "structure" in prompt.lower() or "heading" in prompt.lower()
