"""get_map(floor, zoom) — the occupancy map built as the agent moves (floor=k views another
storey; zoom=true adds an 8 m close-up). Free."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.slam.notes import decode, listen, live_png

NAME = "get_map"
DESCRIPTION = (
    "Read the occupancy map built automatically as you move\n"
    "(it already contains an initial 360-degree scan of your start location).\n"
    "Returns a top-down image cropped to what you have explored — UP on the image\n"
    "is your STARTING heading (+z), right is +x. On it: white = walked-free\n"
    "space, black = obstacle, gray = unexplored; the blue ARROW is you, pointing\n"
    'your current heading; the blue LINE is your path so far; the "S" circle is\n'
    "your start; orange dots are markers you dropped with mark(). Gridlines\n"
    "every 2 m are labeled on both axes with the same x/z coordinates get_pose()\n"
    "reports (the 0-axes are drawn heavier), and the crop window never pans or\n"
    "rescales between calls — it only grows in 2 m steps as you explore, so\n"
    "coordinates stay put from map to map. By default the map shows YOUR CURRENT\n"
    'FLOOR; a "floor k/n" tag (top right) says which storey you are on and how\n'
    "many have been seen. Pass floor=k (the tag's 1-based number) to view\n"
    "another storey's layer, or zoom=true to get TWO images — the full map plus\n"
    "a fixed 8 m close-up centered on you (current floor, same coordinate\n"
    "lattice) — useful when the full map has grown too coarse for nearby\n"
    "detail. Free — no step cost. SLAM estimates, not ground truth."
)


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    listen(s)

    @mcp.tool(description=description)
    def get_map(floor: int = 0, zoom: bool = False) -> list:
        outputs = s.call(
            "get_map",
            timeout=600,
            floor=int(floor) if floor else None,
            view="local" if zoom else "global",
        )
        if outputs.get("map") is None:
            return [json.dumps(outputs.get("info") or {"error": "no map"})]
        png = decode(outputs["map"])
        payload = dict(outputs.get("info") or {})
        live_png(s, png, "map")
        s.live_log({"get_map": {"zoom": bool(zoom)}})
        if not outputs.get("map_local"):
            return [Image(data=png, format="png"), json.dumps(payload)]
        png_l = decode(outputs["map_local"])
        live_png(s, png_l, "map_zoom")
        return [
            "full map:",
            Image(data=png, format="png"),
            "8 m close-up around you:",
            Image(data=png_l, format="png"),
            json.dumps(payload),
        ]
