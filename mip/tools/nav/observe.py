"""observe() — the egocentric RGB view (base64 PNG passed through as an MCP image)."""

from __future__ import annotations

from typing import Any

from mip.core.toolset import Session
from mip.tools.nav.view import capture_view, check_task

NAME = "observe"
DESCRIPTION = (
    "Look through the robot's forward-facing camera. Returns the current egocentric RGB "
    "view. Pure read — does not advance the simulator or consume step budget."
)


def describe(slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    """Slot ``note``: one more sentence after the description."""
    note = slots.get("note")
    return f"{DESCRIPTION} {note}" if note else DESCRIPTION


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    check_task(s)
    # beside other families' tools (a waypoint look, the SLAM instruments) the live log
    # records each primitive look
    two_looks = any(not ref.startswith("nav.") for ref in s.tools)

    @mcp.tool(description=description)
    def observe() -> list:
        s.tool_calls += 1
        png = capture_view(s)
        if two_looks:
            s.live_log({"observe": True})
        return [Image(data=png, format="png")]
