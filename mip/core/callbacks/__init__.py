"""Recorders that plug into the runner (see mip.core.callbacks.base for the
hook contract). The default set, in firing order:

    CodeSnapshot   code/ + code_state.json + provenance      (fatal)
    JsonlLogger    episode_{i}.jsonl
    RawLogger      raw/episode_{i}.jsonl
    PoseProbe      "pose" events                             (env servers with a pose call)
    CostLogger     agent.cost, summary.cost, budget fuse
    VramSampler    run_stats.vram_peak_mib
    StatsReport    stats.json / stats.html

Which set a run gets is the experiment file's ``run.callbacks`` list — one
``{_target_: …}`` entry per recorder, instantiated by runner.py with
hydra.utils.instantiate, in order (PoseProbe only where the env server
exposes a pose call). Adding a recorder = one YAML line; a new recorder = one
Callback subclass + its _target_.
"""

from __future__ import annotations

from typing import Any

from mip.core.callbacks.base import Callback, CallbackSet, StopRun
from mip.core.callbacks.cost import CostLogger
from mip.core.callbacks.jsonl import JsonlLogger
from mip.core.callbacks.pose import PoseProbe
from mip.core.callbacks.raw import RawLogger
from mip.core.callbacks.snapshot import CodeSnapshot
from mip.core.callbacks.stats import StatsReport
from mip.core.callbacks.vram import VramSampler

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
