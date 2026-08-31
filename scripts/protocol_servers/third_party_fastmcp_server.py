import asyncio

from fastmcp import Context, FastMCP
from fastmcp.utilities.tasks import TaskConfig
from fastmcp_tasks import TasksExtension
from mcp.types import ToolAnnotations

mcp = FastMCP(
    name="Protocol Test (third-party fastmcp)",
    instructions="Protocol-only test server.",
)
mcp.add_extension(TasksExtension())


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    task=TaskConfig(mode="optional"),
)
async def progress_demo(
    steps: int = 5,
    delay_ms: int = 200,
    ctx: Context | None = None,
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
