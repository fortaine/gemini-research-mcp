#!/usr/bin/env python3
"""LangChain interoperability smoke test for gemini-research-mcp.

This script proves that gemini-research-mcp can be consumed as an external MCP
server by LangChain's `langchain.mcp.MCPAdapter` (LangChain `1.4.0a2`), over
both the legacy stdio transport and the modern streamable-http transport.

IMPORTANT: LangChain is intentionally NOT a project dependency, dev
dependency, or extra of gemini-research-mcp (see plan.md, chantier 9,
decision 9). This script must always be run from an ephemeral uv
environment that installs LangChain on the fly and discards it afterwards:

    uv run --with "langchain[mcp]==1.4.0a2" python scripts/langchain_interop_smoke.py

Never add `langchain` to pyproject.toml because of this script.

What this proves:
  1. stdio transport: MCPAdapter can launch `uv run gemini-research-mcp` as a
     subprocess, discover the BM25-compacted tool catalog, call a hidden
     utility tool via the `call_tool` proxy, and receive a structured tool
     error for an invalid call.
  2. streamable-http transport: the same, but against a locally running
     streamable-http instance (sessionless, no sticky session).
  3. BM25 discovery: `search_tools` surfaces a hidden utility tool by name.
  4. Elicitation capability declaration: constructing `MCPAdapter(target,
     elicitation="interrupt")` declares the elicitation capability on the
     wire (this script does not exercise a full LangGraph interrupt/resume
     cycle - that requires a LangGraph checkpointer runtime, out of scope
     for a standalone smoke script; see the docstring of
     `langchain.mcp.MCPAdapter` for the full interrupt-mode contract).

Requires GEMINI_API_KEY to be set (the underlying server tools need it to
construct their genai client), but this script deliberately avoids calling
any tool that makes a real, billed Gemini API request - only cost-free
utility tools and protocol-level behavior are exercised.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


async def _check_catalog(adapter: object, label: str) -> None:
    from langchain.mcp import MCPAdapter

    assert isinstance(adapter, MCPAdapter)
    tools = await adapter.get_tools()
    names = sorted(t.name for t in tools)
    print(f"[{label}] discovered {len(names)} LangChain tools: {names}")

    # The BM25-compacted catalog exposes 7 tools: 5 pinned always-visible
    # tools + the synthetic search_tools/call_tool pair.
    expected_always_visible = {
        "research_web",
        "research_deep",
        "research_deep_max",
        "resume_research",
        "export_research_session",
    }
    assert expected_always_visible.issubset(set(names)), names
    assert "search_tools" in names, names
    assert "call_tool" in names, names
    assert len(names) == 7, f"expected a compact 7-tool catalog, got {len(names)}: {names}"

    call_tool = next(t for t in tools if t.name == "call_tool")
    search_tools = next(t for t in tools if t.name == "search_tools")

    # BM25 discovery: a hidden utility tool should be findable by relevance.
    found = await search_tools.ainvoke({"query": "list available export format templates"})
    print(f"[{label}] search_tools('list available export format templates') -> {found!r}")
    assert "list_format_templates" in str(found), found

    # Hidden tools remain directly callable via the call_tool proxy, with no
    # billed Gemini API call required.
    result = await call_tool.ainvoke({"name": "list_format_templates", "arguments": {}})
    print(f"[{label}] call_tool(list_format_templates) -> {str(result)[:200]}")

    # A structured tool error: calling an unknown tool name must fail
    # cleanly (either raised, or surfaced as a structured error artifact)
    # rather than hang or silently succeed.
    try:
        unknown_result = await call_tool.ainvoke(
            {"name": "this_tool_does_not_exist", "arguments": {}}
        )
    except Exception as exc:  # noqa: BLE001 - we want to show whatever error shape surfaces
        print(
            f"[{label}] call_tool(unknown tool) raised structured error: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        text = str(unknown_result)
        print(f"[{label}] call_tool(unknown tool) returned structured error artifact: {text[:200]}")
        assert "error" in text.lower() or "unknown tool" in text.lower(), (
            f"expected an error indication for an unknown tool call, got: {text}"
        )


async def _check_elicitation_capability_declaration() -> None:
    from fastmcp import Client as FastMCPClient
    from langchain.mcp import MCPAdapter

    server_module = REPO_ROOT / "src" / "gemini_research_mcp" / "server.py"
    async with MCPAdapter(str(server_module), elicitation="interrupt") as adapter:
        client = adapter.client
        assert isinstance(client, FastMCPClient)
        # Per MCPAdapter's contract: declaring elicitation="interrupt" wires
        # an elicitation handler onto the underlying client, which is what
        # advertises the elicitation capability to the server during
        # initialize(). We only assert the capability was wired - a full
        # interrupt()/resume cycle needs a LangGraph checkpointer runtime.
        assert client._elicitation_callback is not None  # noqa: SLF001 - inspecting wiring for this smoke test
        print("[elicitation] MCPAdapter(elicitation='interrupt') wired an elicitation handler")


async def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set; server tool construction may fail.", file=sys.stderr)

    from langchain.mcp import MCPAdapter

    print("=== stdio transport ===")
    from fastmcp.client.transports import StdioTransport

    stdio_transport = StdioTransport(
        command="uv",
        args=["run", "--project", str(REPO_ROOT), "gemini-research-mcp"],
    )
    async with MCPAdapter(stdio_transport) as adapter:
        await _check_catalog(adapter, "stdio")

    print("\n=== streamable-http transport ===")
    from fastmcp import Client as FastMCPClient

    from gemini_research_mcp.server import mcp as server_instance

    host, port, path = "127.0.0.1", 8935, "/mcp"
    server_task = asyncio.create_task(
        server_instance.run_async(
            transport="streamable-http",
            host=host,
            port=port,
            path=path,
            stateless_http=True,
            show_banner=False,
        )
    )
    try:
        await asyncio.sleep(0.75)
        url = f"http://{host}:{port}{path}"
        async with MCPAdapter(FastMCPClient(url)) as adapter:
            await _check_catalog(adapter, "streamable-http")

            # Sessionless proof: a second, independent adapter connection
            # must work with no sticky-session state carried over.
            async with MCPAdapter(FastMCPClient(url)) as adapter_b:
                tools_b = await adapter_b.get_tools()
                assert len(tools_b) == 7
                print(
                    "[streamable-http] second independent connection "
                    "also sees 7 tools (sessionless)"
                )
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task

    print("\n=== elicitation capability declaration ===")
    await _check_elicitation_capability_declaration()

    print("\nAll LangChain interop checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
