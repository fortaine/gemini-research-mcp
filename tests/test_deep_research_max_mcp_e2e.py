"""Manual provider diagnostic for Deep Research Max remote MCP.

This test is intentionally excluded from release gates. The product-facing
integration fails closed before provider access. Run this diagnostic only to
evaluate a documented upstream correction:

    RUN_UNSUPPORTED_REMOTE_MCP_DIAGNOSTIC=1 \
    GEMINI_API_KEY=... \
    GEMINI_MCP_REALISTIC_E2E_URL=https://fixture.example/mcp \
    uv run pytest -m e2e tests/test_deep_research_max_mcp_e2e.py -q -s
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from google import genai

from gemini_research_mcp.types import DeepResearchAgent

pytestmark = pytest.mark.e2e

TOOL_NAMES = [
    "market_get_mission",
    "market_get_runtime_policy",
    "market_get_evidence_ledger",
    "market_generate_report",
]
POLL_INTERVAL_SECONDS = 10
MAX_POLLS = 120


def _require_manual_diagnostic() -> tuple[str, str]:
    if os.environ.get("RUN_UNSUPPORTED_REMOTE_MCP_DIAGNOSTIC") != "1":
        pytest.skip("unsupported remote MCP diagnostic is not explicitly enabled")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    mcp_url = os.environ.get("GEMINI_MCP_REALISTIC_E2E_URL")
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    if not mcp_url:
        pytest.skip("GEMINI_MCP_REALISTIC_E2E_URL not set")
    return api_key, mcp_url


async def _verify_fixture(mcp_url: str) -> None:
    async with Client(mcp_url) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed}
        assert set(TOOL_NAMES).issubset(names)
        for name in TOOL_NAMES:
            result = await client.call_tool(name, {})
            assert not result.is_error
            assert result.content


async def _poll_until_complete(client: genai.Client, interaction_id: str) -> Any:
    for _ in range(MAX_POLLS):
        interaction = await client.aio.interactions.get(id=interaction_id)
        status = getattr(interaction, "status", None)
        if status == "completed":
            return interaction
        if status in {"failed", "cancelled", "canceled"}:
            error = getattr(interaction, "error", "unknown provider error")
            raise AssertionError(f"Deep Research Max ended with status={status}: {error}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Deep Research Max did not complete after {MAX_POLLS} polls")


def _write_diagnostic(interaction: Any) -> None:
    artifact_dir = os.environ.get("GEMINI_MCP_E2E_ARTIFACT_DIR")
    if not artifact_dir:
        return

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "native-max-mcp-structured-steps.json"
    output_path.write_text(
        json.dumps(interaction.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_provider_retains_structured_mcp_calls_and_results() -> None:
    api_key, mcp_url = _require_manual_diagnostic()
    await _verify_fixture(mcp_url)

    client = genai.Client(api_key=api_key)
    interaction = await client.aio.interactions.create(
        input=(
            "Call every allowed MCP tool before answering. Use only their exact "
            "Project Saffron Harbor evidence and do not invent substitutes."
        ),
        agent=DeepResearchAgent.DEEP_RESEARCH_MAX.value,
        background=True,
        tools=[
            {
                "type": "mcp_server",
                "name": "realistic_fixture",
                "url": mcp_url,
                "allowed_tools": [{"tools": TOOL_NAMES}],
            }
        ],
    )

    final = await _poll_until_complete(client, str(interaction.id))
    steps = getattr(final, "steps", None) or []
    call_steps = [step for step in steps if getattr(step, "type", None) == "mcp_server_tool_call"]
    result_steps = [
        step for step in steps if getattr(step, "type", None) == "mcp_server_tool_result"
    ]
    _write_diagnostic(final)

    assert {step.name for step in call_steps} == set(TOOL_NAMES)
    assert {step.name for step in result_steps} == set(TOOL_NAMES)
    assert all(getattr(step, "result", None) for step in result_steps)
