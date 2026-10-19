"""get_pose() — the SLAM-estimated pose (x, z, yaw_deg) in the start-anchored frame. Free."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.slam.notes import listen

NAME = "get_pose"
DESCRIPTION = (
    "Read the robot's SLAM-estimated pose: position (x, z) in meters and\n"
    "    heading yaw_deg, in a fixed frame anchored at your start pose (x = right\n"
    "    and z = forward OF YOUR STARTING POSE; yaw_deg is 0 at your starting\n"
    "    heading and INCREASES when you turn right). The same x/z coordinates\n"
    "    label the gridlines on get_map(), so pose numbers place you on the map.\n"
    '    Free — no step cost. slam="slam_lost" means the estimate is currently\n'
    "    stale."
)


def register(mcp: Any, s: Session, description: str) -> None:
    listen(s)

    @mcp.tool(description=description)
    def get_pose() -> list:
        r = s.call("get_pose", timeout=600)["pose"]
        s.live_log({"get_pose": r})
        return [json.dumps(r)]
