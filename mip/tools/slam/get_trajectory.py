"""get_trajectory() — the path walked on this task as (x, z) points, oldest first. Free."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.slam.notes import listen

NAME = "get_trajectory"
DESCRIPTION = (
    "Read the path you have walked ON THIS TASK as (x, z) points in the same\n"
    "    fixed frame as get_pose(), oldest first. Useful to check whether you are\n"
    "    circling or which areas you already covered since this goal began. The map\n"
    "    keeps the whole visit; this is just the current stretch. Free — no step\n"
    "    cost."
)


def register(mcp: Any, s: Session, description: str) -> None:
    listen(s)

    @mcp.tool(description=description)
    def get_trajectory() -> list:
        r = s.call("get_trajectory", timeout=600)["trajectory"]
        s.live_log({"get_trajectory": {"n_total": r.get("n_total")}})
        return [json.dumps(r)]
