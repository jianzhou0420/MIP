"""The SLAM sets' step listener: a ``slam_note`` on a SLAM health transition (lost /
recovered, re-anchored), added to the step / goto result, and the collision facts in the live
log. Every SLAM tool registers it (once per session). The note's dash is "—", or "--" under the
``legacy_dashes`` switch (the SLAM sets' wording)."""

from __future__ import annotations

import base64
from typing import Any

from mip.core.toolset import Session, StepListener


class SlamNotes(StepListener):
    def __init__(self, dash: str = "—") -> None:
        self._dash = dash
        self._last_slam = "ok"
        self._last_reanchors = 0
        self._note: str | None = None
        self._collided = False
        self._collision_count = 0

    def begin(self) -> None:
        self._note, self._collided, self._collision_count = None, False, 0

    def observe(self, info: dict[str, Any]) -> None:
        if info.get("collision"):
            self._collided = True
        self._collision_count = int(info.get("collision_count") or 0)
        self._note = self._transition(info) or self._note

    def result_fields(self) -> dict[str, Any]:
        return {"slam_note": self._note} if self._note else {}

    def log_fields(self, tool: str) -> dict[str, Any]:
        if tool == "goto":
            return {"collision_count": self._collision_count}
        return {"collision": self._collided}

    def _transition(self, info: dict[str, Any]) -> str | None:
        slam_now = str(info.get("slam") or "")
        re_now = int(info.get("reanchors") or 0)
        note: str | None = None
        if slam_now:
            if slam_now == "slam_lost" and self._last_slam == "ok":
                note = (
                    f"SLAM tracking lost {self._dash} get_pose/get_map estimates are stale until "
                    "it recovers."
                )
            elif slam_now == "ok" and self._last_slam == "slam_lost":
                note = "SLAM tracking recovered" + (
                    f" and re-anchored {self._dash} the map reference frame may have shifted "
                    "slightly."
                    if re_now > self._last_reanchors
                    else "."
                )
            self._last_slam = slam_now
            self._last_reanchors = max(self._last_reanchors, re_now)
        return note


def listen(s: Session) -> None:
    """Register the session's SlamNotes, once."""
    if "slam_notes" not in s.notes:
        s.notes["slam_notes"] = SlamNotes("--" if s.options.get("legacy_dashes") else "—")
        s.step_listeners.append(s.notes["slam_notes"])


def live_png(s: Session, png: bytes, suffix: str) -> None:
    """An instrument image on the live counter (numbered obs_ files keep sorted order ==
    emission order)."""
    s.obs_count += 1
    s.live_write(f"obs_{s.obs_count:04d}_{suffix}.png", png, latest=False)


def decode(b64: str) -> bytes:
    return base64.b64decode(b64)
