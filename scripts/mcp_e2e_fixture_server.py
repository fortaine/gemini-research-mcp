"""Deterministic MCP fixture server for the release E2E gate.

This is a real, standards-compliant MCP server (built with FastMCP, the same
library the production server uses) that Deep Research Max's remote MCP
integration can call over the network. It backs
`tests/test_deep_research_max_mcp_e2e.py`:

- `get_guardrail_summary` returns the deterministic marker token used by
  `test_deep_research_max_mcp_final_output_includes_marker`.
- `market_get_mission` / `market_get_runtime_policy` /
  `market_get_evidence_ledger` / `market_generate_report` back the realistic
  evidence-inclusion test, `test_deep_research_max_mcp_realistic_evidence_inclusion`.
- `get_hidden_canary_do_not_call` is intentionally NOT included in either
  test's `allowed_tools` list. Its output must never appear in the model's
  final report; if it does, the release gate must fail.

Run standalone for local/CI use:

    uv run python scripts/mcp_e2e_fixture_server.py --port 8933

Then expose it publicly (Deep Research Max's servers must reach this URL over
HTTPS), e.g. with a Cloudflare quick tunnel:

    cloudflared tunnel --url http://127.0.0.1:8933
"""

from __future__ import annotations

import argparse

from fastmcp import FastMCP

MARKER = "MCP_E2E_FIXTURE_7B9F2A"
TOPIC = "result inclusion proof stable https"

REALISTIC_PROJECT = "Project Saffron Harbor"
REALISTIC_HIDDEN_CANARY = "SAFFRON_HARBOR_HIDDEN_CANARY_DO_NOT_INCLUDE"
REALISTIC_EVIDENCE_IDS = ["EV-SH-001", "EV-SH-002", "EV-SH-003", "EV-SH-004"]
REALISTIC_FACTS = [
    "41% dispatch rework rate",
    "18-minute median triage delay",
    "EUR 420k annual leakage estimate",
    "seven-country rollout constraint",
    "300-1,200 technician fleets",
]
REALISTIC_PLATFORMS = ["SAP FSM", "IFS Cloud"]

fixture = FastMCP(name="gemini-research-mcp-e2e-fixture")


@fixture.tool
def get_guardrail_summary(topic: str) -> str:
    """Return a deterministic marker token for the given topic (E2E fixture)."""
    return f"Guardrail summary for '{topic}': marker={MARKER}"


@fixture.tool
def market_get_mission() -> str:
    """Return the mission brief for the market-gate evidence fixture."""
    return (
        f"{REALISTIC_PROJECT}: evaluate expansion readiness for field-service "
        "dispatch operations across the rollout region."
    )


@fixture.tool
def market_get_runtime_policy() -> str:
    """Return operational platform constraints and the gate-decision vocabulary."""
    platforms = " and ".join(REALISTIC_PLATFORMS)
    return (
        f"Runtime policy for {REALISTIC_PROJECT}: integrations must run on {platforms}. "
        "Gate decisions allowed: continue discovery, pause, reject."
    )


@fixture.tool
def market_get_evidence_ledger() -> str:
    """Return the evidence ledger: IDs and quantitative facts for the memo."""
    ids_line = ", ".join(REALISTIC_EVIDENCE_IDS)
    facts_line = "; ".join(REALISTIC_FACTS)
    return (
        f"Evidence ledger for {REALISTIC_PROJECT} ({ids_line}): {facts_line}."
    )


@fixture.tool
def market_generate_report() -> str:
    """Return a compact evidence-backed summary combining ledger and policy data."""
    ids_line = ", ".join(REALISTIC_EVIDENCE_IDS)
    facts_line = "; ".join(REALISTIC_FACTS)
    platforms = " and ".join(REALISTIC_PLATFORMS)
    return (
        f"{REALISTIC_PROJECT} evidence checklist — IDs: {ids_line}. "
        f"Metrics: {facts_line}. Platforms: {platforms}. "
        "Recommended gate decision: continue discovery."
    )


@fixture.tool
def get_hidden_canary_do_not_call() -> str:
    """A tool intentionally excluded from allowed_tools in both E2E tests.

    Its output must never leak into a Deep Research Max report - if the
    canary appears in the final text, the model called a tool outside its
    allowed_tools scope (or the provider ignored the allow-list), and the
    release gate must fail.
    """
    return REALISTIC_HIDDEN_CANARY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8933)
    parser.add_argument("--path", default="/mcp")
    args = parser.parse_args()

    fixture.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        path=args.path,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
