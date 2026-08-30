import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer(
    name="Protocol Test (official mcp.server.mcpserver)",
    instructions="Protocol-only test server.",
)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def progress_demo(
    steps: int = 5,
    delay_ms: int = 200,
    ctx: Any = None,
) -> str:
    if ctx is None:
        return "ctx missing"

    for i in range(1, steps + 1):
        msg = f"step {i}/{steps}"
        await ctx.report_progress(progress=i, total=steps, message=msg)
        await ctx.info(msg)
        await asyncio.sleep(delay_ms / 1000)

    return "done"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
