"""Recorders that plug into the runner (see core.callbacks.base for the
hook contract). The default set, in firing order:

    CodeSnapshot   code/ + code_state.json + provenance      (fatal)
    JsonlLogger    episode_{i}.jsonl
    RawLogger      raw/episode_{i}.jsonl
    PoseProbe      "pose" events                             (env_habitat only)
    CostLogger     agent.cost, summary.cost, budget fuse
    VramSampler    run_stats.vram_peak_mib
    StatsReport    stats.json / stats.html

Which set a run gets is the experiment file's ``run.callbacks`` list — one
``{_target_: …}`` entry per recorder, instantiated by runner.py with
hydra.utils.instantiate, in order (the env_habitat agents carry PoseProbe,
the others not). Adding a recorder = one YAML line; a new recorder = one
Callback subclass + its _target_.
"""

from __future__ import annotations

from typing import Any

from core.callbacks.base import Callback, CallbackSet, StopRun
from core.callbacks.cost import CostLogger
from core.callbacks.jsonl import JsonlLogger
from core.callbacks.pose import PoseProbe
from core.callbacks.raw import RawLogger
from core.callbacks.snapshot import CodeSnapshot
from core.callbacks.stats import StatsReport
from core.callbacks.vram import VramSampler

__all__ = [
    "Callback",
    "CallbackSet",
    "CodeSnapshot",
    "CostLogger",
    "JsonlLogger",
    "PoseProbe",
    "RawLogger",
    "StatsReport",
    "StopRun",
    "VramSampler",
]
