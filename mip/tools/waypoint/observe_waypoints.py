"""observe_waypoints() — the waypoint look beside the primitive observe/step (the Hybrid set):
the same panorama and numbered options as the Waypoint set's observe(); goto() picks from them."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.waypoint.panorama import DESCRIPTION, NEEDS_WP_SERVER, arm_on_moves, render_and_predict

__all__ = ["DESCRIPTION", "NAME", "NEEDS_WP_SERVER", "register"]
NAME = "observe_waypoints"


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    arm_on_moves(s)

    @mcp.tool(description=description)
    def observe_waypoints() -> list:
        s.tool_calls += 1
        if s.episode_over:
            return [f"episode already over ({s.end_reason}); no more moves possible"]
        png, status = render_and_predict(
            s,
            remaining=True,
            empty_note=(
                "no reachable waypoints predicted here; if you believe you are at "
                "the goal call stop(), otherwise make another move"
            ),
            log_key="observe_waypoints",
        )
        return [Image(data=png, format="png"), json.dumps(status)]
