"""The per-episode and per-run contexts every part of the runner shares.

- ``EpisodeContext``: everything a harness adapter needs for ONE clean
  session (briefing, prompts, budgets, the bridge to spawn and its
  environment). Built by the runner after the env placed the episode.
- ``SessionOutcome``: what one session hands back (the adapter contract
  itself is ``mip.harnesses.base.HarnessAdapter``).
- ``ModelSpec``: the model as the adapters' ``prepare()`` sees it (id + API
  params — the ``agent.model`` block of the four-layer config).
- ``RunContext``: the run as the callbacks see it (name, the full config,
  the flat run cfg, run_dir, the records so far, run-level stats and
  provenance).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mip.core.bus import EventBus

EventSink = EventBus  # the adapters' sink type: harnesses.* import it from here


@dataclass(frozen=True)
class ModelSpec:
    """``agent.model`` as the harness adapters read it once per run: the
    harness-facing id, the run's effort (the top-level ``effort`` key — the
    vendor's own word, ``default`` = send nothing), and the API routing
    params beside the id (api_base / api_gateway / gateway_route …)."""

    model_id: str
    effort: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def extra_dict(self) -> dict[str, Any]:
        return dict(self.extra)


@dataclass
class EpisodeContext:
    """One episode as the harness adapter runs it: who it is (index), what the agent is told
    (the tool set's episode prompts), its budgets, where it runs (the env server, the
    directories), the model, and the tool set whose bridge the session spawns."""

    index: int
    instruction: str  # the briefing's {instruction} slot: the statement
    briefing: str  # rendered system prompt / user-message briefing
    first_prompt: str
    server_url: str
    model: str
    max_turns: int
    step_budget: int
    episode_timeout: int
    workdir: Path
    live_dir: Path
    raw_dir: Path
    effort: str = (
        "default"  # the run's reasoning effort, the vendor's word; "default" = the API's own
    )
    wp_server_url: str = ""  # waypoint-predictor auto_host (Waypoint / Hybrid interfaces)
    benchmark: str = "r2r"  # the task key (task_key); the fake harness reads it
    tool_set: Any = None  # the experiment's mip.core.toolset.ToolSet — bridge / prompts / switches
    max_budget_usd: float | None = None  # per-episode USD fuse (sdk harness only;
    # the CLI's --max-budget-usd)
    # what the env placed, as the bridge is started with it: the task word the tools shape
    # themselves by and the line's action-table size (None: the tool's own default)
    task_word: str = "nav"  # nav | objnav | eqa
    n_actions: int | None = None  # 4 on the paper's R2R / RxR protocol, 6 STANDARD
    extra: dict[str, Any] = field(default_factory=dict)  # harness-specific knobs
    run_dir: Path | None = None  # the run this episode belongs to
    bus: Any = None  # the episode's EventBus (set by the runner)

    def knob(self, name: str, default=None):
        """One switch of the tool set (agent.tools.*: wp_max_moves / instruments …),
        ``default`` when the set does not declare it."""
        if self.tool_set is None:
            return default
        return self.tool_set.switch(name, default)

    @property
    def turn_budget(self) -> int:
        """Bridge broadcast/STOP-gate budget: off (0) in the bare condition."""
        return 0 if self._tools().bare else self.max_turns

    @property
    def bridge_path(self) -> Path:
        """The stdio bridge the session talks to: mip/bridge.py."""
        return self._tools().bridge_path

    def bridge_env(self) -> dict[str, str]:
        """Env for the harnesses that spawn the stdio bridge, rendered by the
        tool set from this context — the driver knows no tool's knobs."""
        return self._tools().render_env(self)

    def _tools(self) -> Any:
        if self.tool_set is None:
            raise RuntimeError(f"episode {self.index} ({self.benchmark}) has no tool set")
        return self.tool_set


@dataclass
class SessionOutcome:
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    turns: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """One run, as the callbacks see it."""

    name: str  # run.name
    app: dict[str, Any]  # the full four-layer config, resolved (task · env · agent · run)
    cfg: dict[str, Any]  # the flat run cfg (mip.core.run.run_cfg)
    run_dir: Path
    servers: list[str]
    adapter: Any
    tool_set: Any  # the agent's interface (mip.core.toolset.ToolSet)
    env_name: str = ""  # env._target_ class name
    episodes_spec: str | None = None
    prior_episodes: dict[int, dict[str, Any]] = field(default_factory=dict)  # resume
    episodes: dict[int, dict[str, Any]] = field(default_factory=dict)  # records so far
    run_stats: dict[str, Any] = field(default_factory=dict)  # summary.run_stats
    provenance: dict[str, Any] = field(default_factory=dict)  # summary.provenance
    summary_extra: dict[str, Any] = field(default_factory=dict)  # run-level keys callbacks add
    callback_errors: dict[str, str] = field(default_factory=dict)
    callbacks: Any = None  # the CallbackSet (set by the runner)
    stopped_by: str | None = None  # the callback that raised StopRun
    drain: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"
