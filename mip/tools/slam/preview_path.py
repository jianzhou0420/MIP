"""preview_path(x, z) — a read-only A* route preview over the map's walked-free cells to
(x, z), drawn on the map. Free."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.slam.notes import decode, listen, live_png

NAME = "preview_path"
DESCRIPTION = (
    "Preview a walkable route from you to map point (x, z) — the same\n"
    "    x/z coordinates get_pose() and the map gridlines use. Plans over the\n"
    "    walked-free (white) cells of your current floor's map and returns the\n"
    "    map image with the route drawn in magenta (X = target) plus the route\n"
    "    length in meters. If (x, z) is not on known free space it snaps to the\n"
    "    nearest free cell within 1 m, or reports it unreachable — the map only\n"
    "    knows space you have already seen. NOTHING MOVES: this is a read-only\n"
    "    preview. Free — no step cost."
)


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    listen(s)

    @mcp.tool(description=description)
    def preview_path(x: float, z: float) -> list:
        outputs = s.call("preview_path", timeout=600, x=x, z=z)
        r = outputs.get("result") or {}
        s.live_log(
            {
                "preview_path": {
                    "x": x,
                    "z": z,
                    "reachable": r.get("reachable"),
                    "distance_m": r.get("distance_m"),
                }
            }
        )
        if outputs.get("map") is None:
            return [json.dumps(r)]
        png = decode(outputs["map"])
        live_png(s, png, "preview")
        return [Image(data=png, format="png"), json.dumps(r)]
