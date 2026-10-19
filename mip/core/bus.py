"""EventBus — the one channel every session event and raw record goes
through, fanned out to the run's callbacks.

Adapters see it as the ``sink``: ``emit(kind, payload)`` for the curated
vocabulary (thinking / assistant_text / tool_use / tool_result /
system_init / bridge_status / driver_error / result / exit / pose …) and
``raw(rec)`` for the harness-native stream. The bus itself keeps the per-tool call
tally for the episode record and the last step result it can parse off the
stream — the fallback when a seat's bridge wrote no Session report (every record reads the
bridge Session's report first, mip.core.toolset.read_report); everything else is a callback's business
(mip/core/callbacks/: JsonlLogger writes episode_{i}.jsonl, RawLogger raw/…).
"""

from __future__ import annotations

import json
import time
from typing import Any

from mip.core.callbacks.base import CallbackSet


class EventBus:
    def __init__(self, ep: Any, callbacks: CallbackSet) -> None:
        self.ep = ep
        self.callbacks = callbacks
        self._t0 = time.monotonic()
        self.tool_calls: dict[str, int] = {}
        self.last_step_result: dict[str, Any] | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_use":
            name = str(payload.get("name", "")).rsplit("__", 1)[-1]
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        elif kind == "tool_result":
            parsed = self.parse_step_result(payload.get("texts") or [])
            if parsed is not None:
                self.last_step_result = parsed
        elif kind == "exit":
            # mini delivers the episode-ending step result as the exit message
            # content, not as a tool_result
            content = payload.get("content")
            parsed = self.parse_step_result([content] if isinstance(content, str) else [])
            if parsed is not None:
                self.last_step_result = parsed
        rec = {"t": round(self.elapsed, 2), "kind": kind, **payload}
        self.callbacks.fire("on_event", self.ep, kind, rec)

    def raw(self, rec: Any) -> None:
        self.callbacks.fire("on_raw", self.ep, rec)

    @staticmethod
    def parse_step_result(texts: list[str]) -> dict[str, Any] | None:
        """The first JSON dict carrying ``steps_taken_total`` — a bridge's
        step / answer result."""
        for text in texts:
            try:
                data = json.loads(text)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict) and "steps_taken_total" in data:
                return data
        return None
