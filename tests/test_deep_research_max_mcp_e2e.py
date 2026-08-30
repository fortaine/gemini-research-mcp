"""Live E2E test for Deep Research Max remote MCP result inclusion.

Run manually:

    GEMINI_API_KEY=... \
    GEMINI_MCP_E2E_URL=https://fresh-stable-fixture.example/mcp \
    uv run pytest -m e2e tests/test_deep_research_max_mcp_e2e.py -q

The MCP endpoint must expose a read-only `get_guardrail_summary` tool returning
the deterministic marker below.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from google import genai

from gemini_research_mcp.deep import _extract_text_from_interaction
from gemini_research_mcp.types import DeepResearchAgent

pytestmark = pytest.mark.e2e

MARKER = "MCP_E2E_FIXTURE_7B9F2A"
TOOL_NAME = "get_guardrail_summary"
TOPIC = "result inclusion proof stable https"
REALISTIC_PROJECT = "Project Saffron Harbor"
REALISTIC_HIDDEN_CANARY = "SAFFRON_HARBOR_HIDDEN_CANARY_DO_NOT_INCLUDE"
REALISTIC_TOOL_NAMES = [
    "market_get_mission",
    "market_get_runtime_policy",
    "market_get_evidence_ledger",
    "market_generate_report",
]
REALISTIC_EVIDENCE_IDS = ["EV-SH-001", "EV-SH-002", "EV-SH-003", "EV-SH-004"]
REALISTIC_FACTS = [
    "41% dispatch rework rate",
    "18-minute median triage delay",
    "EUR 420k annual leakage estimate",
    "seven-country rollout constraint",
    "300-1,200 technician fleets",
]
REALISTIC_PLATFORMS = ["SAP FSM", "IFS Cloud"]
REALISTIC_DECISIONS = ["continue discovery", "continue-discovery", "pause", "reject"]
POLL_INTERVAL_SECONDS = 10
MAX_POLLS = 90


def _text_outputs(interaction: Any) -> str:
    return _extract_text_from_interaction(interaction) or ""


def _contains_marker_token(text: str) -> bool:
    return MARKER in text or MARKER in "".join(text.split())


def _contains_text(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _count_present(text: str, needles: list[str]) -> int:
    return sum(1 for needle in needles if _contains_text(text, needle))


def _realistic_evidence_check(text: str) -> dict[str, Any]:
    return {
        "has_project": _contains_text(text, REALISTIC_PROJECT),
        "evidence_ids_present": [
            evidence_id
            for evidence_id in REALISTIC_EVIDENCE_IDS
            if _contains_text(text, evidence_id)
        ],
        "facts_present": [fact for fact in REALISTIC_FACTS if _contains_text(text, fact)],
        "platforms_present": [
            platform for platform in REALISTIC_PLATFORMS if _contains_text(text, platform)
        ],
        "has_gate_decision": any(
            _contains_text(text, decision) for decision in REALISTIC_DECISIONS
        ),
        "has_hidden_canary": _contains_text(text, REALISTIC_HIDDEN_CANARY),
    }


def _assert_realistic_evidence_included(text: str) -> dict[str, Any]:
    check = _realistic_evidence_check(text)
    assert check["has_project"], check
    assert len(check["evidence_ids_present"]) >= 3, check
    assert len(check["facts_present"]) >= 3, check
    assert check["platforms_present"], check
    assert check["has_gate_decision"], check
    assert not check["has_hidden_canary"], check
    return check


def _write_realistic_artifacts(
    *,
    interaction_id: str,
    final_text: str,
    check: dict[str, Any],
    final_interaction: Any,
) -> None:
    artifact_dir = os.environ.get("GEMINI_MCP_E2E_ARTIFACT_DIR")
    if not artifact_dir:
        return

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "realistic-deep-research-max-mcp-raw-output.md"
    meta_path = output_dir / "realistic-deep-research-max-mcp-meta.json"

    raw_path.write_text(final_text, encoding="utf-8")
    steps = getattr(final_interaction, "steps", None) or []
    meta_path.write_text(
        json.dumps(
            {
                "interaction_id": interaction_id,
                "status": getattr(final_interaction, "status", None),
                "raw_markdown_path": str(raw_path),
                "evidence_check": check,
                "steps": [
                    {
                        "index": index,
                        "python_type": type(step).__name__,
                        "type": getattr(step, "type", None),
                        "content_count": len(getattr(step, "content", []) or []),
                    }
                    for index, step in enumerate(steps)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_deep_research_max_mcp_final_output_includes_marker() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    mcp_url = os.environ.get("GEMINI_MCP_E2E_URL")
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    if not mcp_url:
        pytest.skip("GEMINI_MCP_E2E_URL not set")

    tools = [
        {
            "type": "mcp_server",
            "name": "fixture_service",
            "url": mcp_url,
            "allowed_tools": [{"tools": [TOOL_NAME]}],
        }
    ]
    prompt = (
        f'Call the MCP tool {TOOL_NAME} with topic "{TOPIC}".\n'
        "The tool returns a marker token containing underscores. Your final answer "
        "must copy that returned marker token exactly as one contiguous token, with "
        "no spaces or line breaks inserted inside it.\n"
        "Return a single short sentence containing the exact marker token and no "
        "Markdown formatting.\n"
        "Do not perform web research. Do not answer from prior knowledge."
    )

    client = genai.Client(api_key=api_key)

    normal_control = await client.aio.interactions.create(
        input=prompt,
        model="gemini-2.5-flash",
        tools=tools,
    )
    assert _contains_marker_token(_text_outputs(normal_control))

    stream = await client.aio.interactions.create(
        input=prompt,
        agent=DeepResearchAgent.DEEP_RESEARCH_MAX.value,
        background=True,
        stream=True,
        agent_config={"type": "deep-research", "thinking_summaries": "auto"},
        tools=tools,
    )

    interaction_id: str | None = None
    async for event in stream:
        if getattr(event, "event_type", None) == "interaction.created":
            interaction_id = str(event.interaction.id)
        elif getattr(event, "event_type", None) == "error":
            raise AssertionError(f"Deep Research Max stream error: {event!r}")

    assert interaction_id, "Deep Research Max did not emit interaction.created"

    final_interaction = await _poll_until_complete(client, interaction_id)
    final_text = _text_outputs(final_interaction)
    assert _contains_marker_token(final_text)


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_deep_research_max_mcp_realistic_evidence_inclusion() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    mcp_url = os.environ.get("GEMINI_MCP_REALISTIC_E2E_URL")
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    if not mcp_url:
        pytest.skip("GEMINI_MCP_REALISTIC_E2E_URL not set")

    tools = [
        {
            "type": "mcp_server",
            "name": "realistic_fixture",
            "url": mcp_url,
            "allowed_tools": [{"tools": REALISTIC_TOOL_NAMES}],
        }
    ]
    prompt = (
        "Start a new evidence-led research memo using only the remote MCP server data.\n"
        "Call these MCP tools before writing the memo: "
        f"{', '.join(REALISTIC_TOOL_NAMES)}.\n"
        "Analyze the market gate for Project Saffron Harbor. Your final report must "
        "cite MCP evidence IDs, include the operational metrics, name relevant platform "
        "constraints, and make a clear gate recommendation. Do not use hidden tools. "
        "Do not invent TAM, revenue, customer counts, or public-web claims."
    )

    client = genai.Client(api_key=api_key)

    normal_control = await client.aio.interactions.create(
        input=(
            "Call the allowed MCP tools and return a compact evidence checklist for "
            "Project Saffron Harbor with evidence IDs, metrics, platforms, and gate decision."
        ),
        model="gemini-2.5-flash",
        tools=tools,
    )
    normal_text = _text_outputs(normal_control)
    assert _count_present(normal_text, REALISTIC_EVIDENCE_IDS) >= 3
    assert _count_present(normal_text, REALISTIC_FACTS) >= 3
    assert not _contains_text(normal_text, REALISTIC_HIDDEN_CANARY)

    stream = await client.aio.interactions.create(
        input=prompt,
        agent=DeepResearchAgent.DEEP_RESEARCH_MAX.value,
        background=True,
        stream=True,
        agent_config={"type": "deep-research", "thinking_summaries": "auto"},
        tools=tools,
    )

    interaction_id: str | None = None
    async for event in stream:
        if getattr(event, "event_type", None) == "interaction.created":
            interaction_id = str(event.interaction.id)
        elif getattr(event, "event_type", None) == "error":
            raise AssertionError(f"Deep Research Max stream error: {event!r}")

    assert interaction_id, "Deep Research Max did not emit interaction.created"

    final_interaction = await _poll_until_complete(client, interaction_id)
    final_text = _text_outputs(final_interaction)
    check = _assert_realistic_evidence_included(final_text)
    _write_realistic_artifacts(
        interaction_id=interaction_id,
        final_text=final_text,
        check=check,
        final_interaction=final_interaction,
    )
