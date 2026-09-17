"""PoseProbe — the env's GT camera pose after every step, as "pose" events.

A side-channel for the env_habitat lines (2026-09-06): after each
step-bearing tool_result the probe reads ``env_habitat__observe_camera_pose``
and the pose lands in episode_{i}.jsonl as its own event — never in the
agent's context, so fair play is untouched. Advisory: the agent may already
have the next call in flight, so a pose can lag the batch by one call.
Self-disables after two consecutive failures so observability can never
take a run down. Registered AFTER JsonlLogger so the pose line follows the
tool_result line it belongs to.
"""

from __future__ import annotations

from typing import Any

import requests

from core.callbacks.base import Callback


class PoseProbe(Callback):
    writes = ("episode_{i}.jsonl:pose",)

    def __init__(self) -> None:
        self._fails: dict[str, int] = {}

    def _probe(self, server_url: str) -> dict[str, Any] | None:
        if self._fails.get(server_url, 0) >= 2:
            return None
        try:
            resp = requests.post(
                f"{server_url}/call/env_habitat__observe_camera_pose",
                json={"inputs": {"trigger": "driver"}}, timeout=5)
            resp.raise_for_status()
            out = resp.json()["outputs"]
            self._fails[server_url] = 0
            return {"position": out.get("position"),
                    "rotation": out.get("rotation")}
        except Exception:
            self._fails[server_url] = self._fails.get(server_url, 0) + 1
            return None

    def on_event(self, ep: Any, kind: str, rec: dict[str, Any]) -> None:
        if kind != "tool_result" or ep.bus is None:
            return
        if ep.bus.parse_step_result(rec.get("texts") or []) is None:
            return
        pose = self._probe(ep.server_url)
        if pose:
            ep.bus.emit("pose", pose)
