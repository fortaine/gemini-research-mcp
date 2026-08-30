"""Test configuration for deep-research MCP server."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires API key)"
    )
    config.addinivalue_line(
        "markers", "redis: mark test as requiring a live Redis/Valkey instance"
    )


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio for async tests."""
    return "asyncio"
