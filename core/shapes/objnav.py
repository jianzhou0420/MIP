"""objnav — ObjectNav: find any instance of a goal category, end by STOP.

Serves env_slam_es's objnav profiles (hm3d / mp3d). reset returns a goal CATEGORY, not an
instruction — ObjectNav episodes carry no language. The recorded
"instruction" is the goal text the briefing embeds (underscores are
dataset-internal: "tv_monitor" -> "tv monitor"). Evaluate is the env's own ruler, driver-triggered.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import Goal, TaskModule


class ObjNav(TaskModule):
    shape = "objnav"
    task_word = "objnav"

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        category = str(ep.get("object_category") or "").strip()
        if not category:
            raise RuntimeError(
                f"{self.env.env_name} reset returned no object_category "
                f"(episode {index}) — episode placement failed, or the env "
                "is not on an objnav dataset?"
            )
        return Goal(text=category.replace("_", " "), task="objnav", ep=ep)

    def bridge_knobs(self, goal: Goal) -> dict[str, Any]:
        # the slam arms' task surface (their bridges default to "nav")
        return {"slam_task": "objnav"}
