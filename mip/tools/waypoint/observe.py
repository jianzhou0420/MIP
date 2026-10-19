"""observe() — the Waypoint set's only look: the 12-view panorama, the predictor's candidates
drawn on the [Left|Front|Right|Back] strip, and a JSON of the numbered options."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.waypoint.panorama import DESCRIPTION, NEEDS_WP_SERVER, arm_on_moves, render_and_predict

__all__ = ["DESCRIPTION", "NAME", "NEEDS_WP_SERVER", "register"]
NAME = "observe"


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    arm_on_moves(s)

    @mcp.tool(description=description)
    def observe() -> list:
        s.tool_calls += 1
        if s.episode_over:
            return [f"episode already over ({s.end_reason}); no more moves possible"]
        png, status = render_and_predict(
            s,
            remaining=False,
            empty_note=(
                "no reachable waypoints predicted here; if you believe you are at "
                "the goal call stop(), otherwise goto() a prior waypoint or make "
                "another move"
            ),
            log_key="observe_wp",
        )
        return [Image(data=png, format="png"), json.dumps(status)]
