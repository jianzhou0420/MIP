"""mark(label) — drop (or move) a named marker at the estimated position; it shows on later
maps. Free."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.slam.notes import listen

NAME = "mark"
DESCRIPTION = (
    "Drop a named marker at your current position — it appears as an\n"
    "    orange dot with your label on every later get_map() (on the floor where\n"
    "    you dropped it), and in the map's JSON. Use it to remember places the\n"
    '    instruction cares about ("kitchen", "blue door") instead of trusting\n'
    "    visual memory. Re-marking an existing label MOVES it to where you are\n"
    "    now. Labels cap at 24 characters. Free — no step cost."
)


def describe(slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    return DESCRIPTION


def register(mcp: Any, s: Session, description: str) -> None:
    listen(s)

    @mcp.tool(description=description)
    def mark(label: str) -> list:
        r = s.call("mark", timeout=600, label=label)["result"]
        s.live_log({"mark": r})
        return [json.dumps(r)]
