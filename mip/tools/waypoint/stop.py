"""stop() — discrete action 0 on the env; permanently ends the episode."""

from __future__ import annotations

from typing import Any

from mip.core.toolset import Session
from mip.tools.waypoint.panorama import ENV_TIMEOUT_S

NAME = "stop"
DESCRIPTION = (
    "Permanently END the episode, declaring you have reached the goal. Issue it only "
    "when you believe the robot is within 3 meters of the instruction's endpoint — "
    "stopping is irreversible."
)


def register(mcp: Any, s: Session, description: str) -> None:
    @mcp.tool(description=description)
    def stop() -> dict[str, Any]:
        s.tool_calls += 1
        if s.episode_over:
            return {"error": f"episode already over ({s.end_reason}); no more moves possible"}
        s.call("step", timeout=ENV_TIMEOUT_S, action=0)
        s.steps_taken += 1
        s.episode_over = True
        s.end_reason = "stop_called"
        result = {
            "stopped": True,
            "steps_taken_total": s.steps_taken,
            "episode_over": True,
            "end_reason": s.end_reason,
        }
        s.live_log(result)
        return result
