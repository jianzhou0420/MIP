"""look_around() — four views 90 degrees apart (front / right / back / left), rendered in
place; SLAM-silent. Free."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.slam.notes import decode, listen, live_png

NAME = "look_around"
DESCRIPTION = (
    "Turn a full circle in place and return four camera views 90 degrees\n"
    "    apart — front, right, back, left relative to your CURRENT heading —\n"
    "    ending faced exactly where you started. The camera's field of view is 90\n"
    "    degrees, so together the four views tile the complete panorama. This is\n"
    "    a pure LOOK: like observe(), it does not touch SLAM or the map —\n"
    "    get_map() stays exactly as it was, and only walking builds the map.\n"
    "    Free — no step cost, no budget; turning in place cannot collide."
)

VIEWS = (
    ("front (your heading, 0°)", "view_front"),
    ("right (+90°)", "view_right"),
    ("back (180°)", "view_back"),
    ("left (-90°)", "view_left"),
)


def describe(slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    return DESCRIPTION


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    listen(s)

    @mcp.tool(description=description)
    def look_around() -> list:
        if s.episode_over:
            return [json.dumps({"error": f"episode already over ({s.end_reason})"})]
        outputs = s.call("look_around", timeout=600)
        info = outputs.get("info") or {}
        if "error" in info:
            return [json.dumps({"error": info["error"]})]
        result: list = []
        for i, (label, port) in enumerate(VIEWS):
            png = decode(outputs[port])
            live_png(s, png, f"pano{i}")
            result.append(label)
            result.append(Image(data=png, format="png"))
        s.live_log({"look_around": {"slam": info.get("slam")}})
        return result
