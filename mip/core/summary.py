"""summary.json — every episode record of a run, the aggregate over the scored ones, and the
run-level keys around them, flushed after every episode.

``Summary`` is the file: ``build`` assembles it from the run (the loop's head fields, the
records, ``run_stats``, ``provenance``, ``writers``, and whatever run-level keys the callbacks
added), ``flush`` writes it, ``absorb`` keeps the keys an ``on_run_end`` hook put into a built
summary, ``prior`` reads a previous launch's records for a resume. The scoring rule
(``is_scored``) and the ``aggregate`` live here because nothing else decides what a run's
reported number is.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mip.core.episode import RunContext

#: the keys the summary owns besides the head; everything else is a callback's
OWN_KEYS = ("run_stats", "aggregate", "episodes", "provenance", "writers")


def is_scored(rec: dict[str, Any]) -> bool:
    """True if the episode is a real, scored navigation attempt that counts
    toward SR.

    INCLUDES a turn-exhausted episode (hit the SDK ``max_turns`` cap without
    calling STOP): it navigated, was evaluated, ``success`` is 0.0 — a
    legitimate FAILURE, not an error to hide (dropping it inflates SR).

    EXCLUDES two kinds of non-attempts:
    - genuine infra failures (timeout / crash) that produced no metrics
      (``metrics == {}`` → ``success is None``);
    - rate-limit / 'limit exceeded' casualties — errored WITHOUT taking a
      single navigation step (``error`` set + ``env_steps`` 0), so the model
      never really attempted the task. (A genuine turn-exhausted run has
      ``env_steps`` > 0 and stays counted.)
    """
    if (rec.get("metrics") or {}).get("success") is None:
        return False
    return not (rec.get("error") and not ((rec.get("agent") or {}).get("env_steps") or 0))


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"episode_count": len(episodes)}
    numeric: dict[str, list[float]] = {}
    for rec in episodes:
        for key, value in (rec.get("metrics") or {}).items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
        numeric.setdefault("env_steps", []).append(float(rec["agent"].get("env_steps", 0)))
    for key, values in numeric.items():
        if values:
            agg[key] = round(sum(values) / len(values), 4)
    agg["stop_rate"] = round(
        sum(1 for r in episodes if r["agent"].get("called_stop")) / max(1, len(episodes)), 4
    )
    return agg


def stub_record(index: int, error: str) -> dict[str, Any]:
    """The record of an episode that never produced one (a timeout, a crash)."""
    return {
        "index": index,
        "error": error,
        "metrics": {},
        "agent": {"env_steps": 0, "called_stop": False, "tool_calls": {}},
    }


class Summary:
    """The run's summary.json. ``head`` gives the loop's own fields (name, harness, config,
    servers …), read at every build because some of them change while the run is on."""

    def __init__(self, run: RunContext, head: Callable[[], dict[str, Any]]) -> None:
        self.run = run
        self.head = head
        self._lock = asyncio.Lock()

    @staticmethod
    def prior(path: Path) -> dict[int, dict[str, Any]]:
        """A previous launch's records by index (a resume keeps those it does not re-run);
        empty when there is none or it does not parse."""
        if not path.exists():
            return {}
        try:
            return {int(rec["index"]): rec for rec in json.loads(path.read_text())["episodes"]}
        except Exception:
            return {}

    def scored(self) -> dict[str, Any]:
        """The aggregate over the scored records so far."""
        return aggregate([e for e in self.run.episodes.values() if is_scored(e)])

    def build(self) -> dict[str, Any]:
        run = self.run
        ordered = [run.episodes[i] for i in sorted(run.episodes)]
        return {
            **self.head(),
            "run_stats": run.run_stats,
            "aggregate": aggregate([e for e in ordered if is_scored(e)]),
            "episodes": ordered,
            **run.summary_extra,
            "provenance": run.provenance,
            "writers": run.callbacks.writers(),
        }

    def absorb(self, summary: dict[str, Any]) -> None:
        """Keep the run-level keys an ``on_run_end`` hook added to a built summary."""
        own = set(self.head()) | set(OWN_KEYS)
        for key, value in summary.items():
            if key not in own:
                self.run.summary_extra[key] = value

    async def flush(self, summary: dict[str, Any] | None = None) -> None:
        async with self._lock:
            self.run.summary_path.write_text(json.dumps(summary or self.build(), indent=2))
