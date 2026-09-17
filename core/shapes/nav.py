"""nav — instruction-following navigation on the std habitat body.

Serves env_slam_es's nav profiles (R2R-CE / RxR-CE). reset takes the
rgb_resolution config and returns the instruction verbatim; evaluate is
the env's own ruler, driver-triggered; the episode ends by STOP.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import Goal, TaskModule


class Nav(TaskModule):
    shape = "nav"
    task_word = "nav"

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        return Goal(text=ep["instruction"], task="nav", ep=ep)

