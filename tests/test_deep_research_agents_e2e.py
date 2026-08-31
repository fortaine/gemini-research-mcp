"""Paid E2E gates for the supported Deep Research agents without remote MCP."""

from __future__ import annotations

import os

import pytest

from gemini_research_mcp.deep import deep_research
from gemini_research_mcp.types import DeepResearchAgent

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "agent_name",
    [
        DeepResearchAgent.DEEP_RESEARCH,
        DeepResearchAgent.DEEP_RESEARCH_MAX,
    ],
)
async def test_deep_research_agent_completes_without_remote_mcp(
    agent_name: DeepResearchAgent,
) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    result = await deep_research(
        (
            "In at most five bullets, explain why deterministic end-to-end "
            "tests are important for an MCP server release."
        ),
        format_instructions="Return at most five concise Markdown bullets.",
        agent_name=agent_name,
        resolve_citations=False,
    )

    assert result.interaction_id
    assert result.text.strip()
    assert "test" in result.text.lower()
