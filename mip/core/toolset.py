"""ToolSet — the interface an experiment assembles from ``mip/tools/`` (what the agent is given: its
tools behind one MCP bridge, the prompts that brief it, the switches), and the bridge side of it.

Runner side, ``ToolSet``: the experiment file names its tools and its prompts::

    defaults:
      - prompts/fragments@prompts.fragments: standard   # mip/prompts/fragments/standard.yaml
      - prompts: nav_4                                  # mip/prompts/nav_4.yaml
    agent:
      tools:
        _target_: mip.core.toolset.ToolSet
        word: minimal                  # the interface's segment of run.name
        use: [nav.observe, nav.step]  # mip/tools/nav/observe.py, mip/tools/nav/step.py — registered in this order
        prompts: ${prompts}
        bare: true                    # switches: any extra key is one (the tools read them as options)
        auto_observe: false

Every tool set runs the one bridge ``mip/core/bridge.py``; ``render_env`` hands it the episode as
one JSON object (``MIP_BRIDGE``, a ``BridgeSpec``). Per episode the loop asks it once for what
the agent is told (``episode_prompts``: the statement, the briefing, the first message).

The prompts are a Hydra group (``mip/prompts/``): ``first_message`` opens the
session, ``system`` is the briefing with the per-episode slots ``{instruction}`` ``{budget}``,
``tools`` gives each tool the task's facts its description leaves open. The briefing is in the
yaml; the tool descriptions are the tool modules' own text.

A tool module carries ``NAME`` (the MCP tool name), its description and ``register``. The
description is the module's own text: ``DESCRIPTION``, a template whose slots the experiment's
prompts group fills from ``tools.<NAME>`` (the task's facts: nav.step's action table), or ``describe(slots, task=, names=)`` where the
text also follows the placement's task word or the set's other tools. ``describe`` below
renders it runner side, so the bridge receives finished text.

Bridge side, ``Session`` and ``build_server``: one bridge process serves one agent session =
one episode. ``build_server`` imports each named tool module and calls its
``register(mcp, session, description)``; every tool shares the ``Session`` — the env server
it calls, the step accounting, the live directory. After every tool call the Session writes
its report (``<live_dir>/session.json``: steps taken, end reason, the answer), which is what the
runner's episode record reads — never the text the agent was shown.
"""

from __future__ import annotations

import abc
import functools
import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from mip.core.paths import REPO_ROOT

BRIDGE = REPO_ROOT / "mip" / "core" / "bridge.py"
SPEC_ENV = "MIP_BRIDGE"  #: the one environment variable the bridge reads
REPORT = "session.json"  #: the Session's report, in the episode's live directory


class BridgeSpec(TypedDict):
    """One episode as the bridge receives it (JSON in ``MIP_BRIDGE``)."""

    tools: list[str]  # tool modules under mip/tools/, in registration order
    descriptions: dict[str, str]  # MCP tool name -> description (rendered by ``describe``)
    server_url: str
    wp_server_url: str | None  # the waypoint predictor (run.wp_server), for the waypoint tools
    task: str  # the placement's task word: nav | objnav | eqa
    n_actions: int | None  # the line's action-table size (None: the tool's default)
    step_budget: int
    turn_budget: int  # >0 turns on the per-call budget broadcast (0 in the bare condition)
    live_dir: str | None
    options: dict[str, Any]  # the tool set's switches


def tool_module(ref: str) -> Any:
    """The module of one tool reference (``nav.step`` -> ``mip.tools.nav.step``)."""
    return importlib.import_module(f"mip.tools.{ref}")


def describe(ref: str, slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    """One tool's description: its module's text with the slots the experiment gives it
    (``slots``), for this placement's task word and a set of tools named ``names``."""
    module = tool_module(ref)
    try:
        if hasattr(module, "describe"):
            return module.describe(dict(slots), task=task, names=names)
        return module.DESCRIPTION.format(**slots)
    except KeyError as missing:
        raise ValueError(
            f"tool {module.NAME!r} ({ref}): no description for {missing} — the experiment's "
            f"prompts group gives tools.{module.NAME} its slots (task {task!r})"
        ) from None


@dataclass(frozen=True)
class EpisodePrompts:
    """What the agent is told in one episode."""

    statement: str  #: the briefing's {instruction} slot
    briefing: str  #: the system prompt, its slots filled
    first_prompt: str  #: the session's opening user message


class ToolSet:
    def __init__(self, word: str, use: list[str], prompts: dict[str, Any], **switches: Any) -> None:
        self.word = word  #: the {word} segment of run.name
        if not isinstance(prompts, dict):
            raise TypeError(
                f"ToolSet({word}): prompts must be the composed prompts group "
                f"(prompts: ${{prompts}}), got {type(prompts).__name__} {prompts!r}"
            )
        self.prompts: dict[str, Any] = dict(prompts)
        self._switches = dict(switches)
        self.use = tuple(use)

    def __repr__(self) -> str:
        return f"ToolSet({self.word})"

    @property
    def name(self) -> str:
        """The tool set's name in the records (``provenance.arm`` of every summary)."""
        return self.word

    # ── switches ──

    def switches(self) -> dict[str, Any]:
        """The switches (bare / auto_observe / wp_max_moves …) as declared."""
        return dict(self._switches)

    def switch(self, name: str, default: Any = None) -> Any:
        return self._switches.get(name, default)

    @property
    def bare(self) -> bool:
        return bool(self._switches.get("bare", False))

    # ── what the agent is told ──

    def episode_prompts(self, placement: dict[str, Any], step_budget: int) -> EpisodePrompts:
        """One episode's statement, briefing (the group's ``system`` with {instruction}
        {budget} filled; nothing else in the text is touched) and first message."""
        statement = self.statement(placement)
        briefing = fill(str(self.prompts["system"]), instruction=statement, budget=step_budget)
        return EpisodePrompts(statement, briefing, str(self.prompts["first_message"]))

    def statement(self, placement: dict[str, Any]) -> str:
        """The ``{instruction}`` slot of one episode: the placement's ``task`` — the
        dataset's own statement, as the env's ``place(index)`` returned it."""
        return str(placement.get("task") or "")

    # ── the tools and their bridge ──

    @property
    def bridge_path(self) -> Path:
        return BRIDGE

    @property
    def needs_wp_server(self) -> bool:
        return any(getattr(tool_module(ref), "NEEDS_WP_SERVER", False) for ref in self.use)

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """The MCP tool names, in registration order (the Claude SDK session records them)."""
        return tuple(tool_module(ref).NAME for ref in self.use)

    def families(self) -> list[Any]:
        """The tool families of the set (``tools.<family>`` packages), in first-use order."""
        names = dict.fromkeys(ref.split(".", 1)[0] for ref in self.use)
        return [importlib.import_module(f"mip.tools.{name}") for name in names]

    def preflight(self, **facts: Any) -> None:
        """Before episode 0: every tool family that checks its own run-level needs (its
        package's ``preflight``: the waypoint predictor, the SLAM hook) does, given the run's
        facts (``wp_server``, ``dataset``, ``env``: the env block); raises to refuse."""
        for family in self.families():
            if (check := getattr(family, "preflight", None)) is not None:
                check(**facts)

    def env_fields(self) -> dict[str, Any]:
        """The switches the set's families push to the env server's configure (their
        package's ``ENV_FIELDS``: slam's instruments), as declared on this set."""
        names = [n for family in self.families() for n in getattr(family, "ENV_FIELDS", ())]
        return {n: self._switches[n] for n in names if n in self._switches}

    def descriptions(self, task: str) -> dict[str, str]:
        """Each tool's description (MCP tool name -> text) for a placement of task word ``task``."""
        slots = self.prompts.get("tools") or {}
        names = self.allowed_tools
        return {
            name: describe(ref, slots.get(name) or {}, task=task, names=names)
            for ref, name in zip(self.use, names, strict=True)
        }

    def render_env(self, ctx: Any) -> dict[str, str]:
        spec: BridgeSpec = {
            "tools": list(self.use),
            "descriptions": self.descriptions(ctx.task_word),
            "server_url": ctx.server_url,
            "wp_server_url": ctx.wp_server_url or None,
            "task": ctx.task_word,
            "n_actions": ctx.n_actions,
            "step_budget": int(ctx.step_budget),
            "turn_budget": int(ctx.turn_budget),
            "live_dir": str(ctx.live_dir) if ctx.live_dir else None,
            "options": dict(self._switches),
        }
        return {SPEC_ENV: json.dumps(spec, ensure_ascii=False)}

    def describe(self) -> dict[str, Any]:
        """The tool set as config.yaml and the summaries record it."""
        return {
            "word": self.word,
            "tools": list(self.use),
            "prompts": dict(self.prompts),
            "switches": dict(self._switches),
        }


def fill(template: str, **slots: Any) -> str:
    """A prompt text with its per-episode slots substituted — plain
    replacement of ``{name}``, not str.format, so every other brace in a
    briefing (a JSON example, ``step([2,2,2])``) stays as written."""
    for name, value in slots.items():
        template = template.replace("{" + name + "}", str(value))
    return template


class StepListener(abc.ABC):
    """A tool family's view of the env steps a move tool takes (step, goto): it reads each
    step's info and may add fields to that tool call's result and to its live-log entry."""

    def begin(self) -> None:  # optional: most listeners keep no per-call state
        """A move tool call starts."""

    @abc.abstractmethod
    def observe(self, info: dict[str, Any]) -> None:
        """One env step's info, as the env returned it."""

    def result_fields(self) -> dict[str, Any]:
        return {}

    def log_fields(self, tool: str) -> dict[str, Any]:
        return {}


class Session:
    """The state every tool of one bridge shares: the episode's env server and budgets, its
    step accounting, and the live directory a spectator reads."""

    def __init__(self, spec: BridgeSpec) -> None:
        self.tools = list(spec["tools"])
        self.server_url = spec["server_url"]
        from mip.core.envclient import EnvClient  # here, not at import: the env server never needs it

        self._env = EnvClient(self.server_url, timeout_s=300)
        self.wp_server_url = spec.get("wp_server_url")
        self.task = spec["task"]
        self.n_actions = spec.get("n_actions")
        self.step_budget = int(spec["step_budget"])
        self.turn_budget = int(spec["turn_budget"])
        self.live_dir = Path(spec["live_dir"]) if spec.get("live_dir") else None
        self.options = dict(spec.get("options") or {})
        self.steps_taken = 0
        self.obs_count = 0
        self.tool_calls = 0
        self.episode_over = False
        self.end_reason: str | None = None
        self.answer: str | None = None
        self.notes: dict[str, Any] = {}  #: a tool family's own episode state (the waypoint moves …)
        self._on_move: list[Callable[[], None]] = []
        self.step_listeners: list[StepListener] = []
        self._t0 = time.time()

    def on_move(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` whenever a tool moves the robot (a waypoint list goes stale)."""
        self._on_move.append(listener)

    def moved(self) -> None:
        for listener in self._on_move:
            listener()

    def call(self, verb: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        """One env verb (mip.core.envclient: POST /<verb>, kwargs as JSON; 300 s unless ``timeout``)."""
        return self._env.call_sync(verb, timeout, **kwargs)

    def calls_remaining(self) -> int | None:
        """Tool calls left under the turn budget; None when the broadcast is off."""
        if self.turn_budget <= 0:
            return None
        return max(0, self.turn_budget - self.tool_calls)

    def live_write(self, name: str, png: bytes, *, latest: bool = True) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        (self.live_dir / name).write_bytes(png)
        if latest:
            (self.live_dir / "latest.png").write_bytes(png)

    def live_log(self, entry: dict[str, Any]) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        with (self.live_dir / "actions.log").open("a") as fh:
            fh.write(json.dumps({"t": round(time.time() - self._t0, 1), **entry}) + "\n")

    def report(self) -> None:
        """How the episode stands, for the runner's record (``read_report``): written whole
        after every tool call, so a session killed mid-episode still leaves its last state."""
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "steps_taken_total": self.steps_taken,
            "episode_over": self.episode_over,
            "end_reason": self.end_reason,
            "answer": self.answer,
        }
        part = self.live_dir / (REPORT + ".part")
        part.write_text(json.dumps(state) + "\n")
        part.replace(self.live_dir / REPORT)


def read_report(live_dir: Path) -> dict[str, Any] | None:
    """The Session's last report of an episode; None when its bridge never ran a tool."""
    try:
        return json.loads((live_dir / REPORT).read_text())
    except (OSError, ValueError):
        return None


class _Reporting:
    """FastMCP as the tool modules see it: every tool they register ends its call with the
    Session's report. The wrapper keeps the tool's signature and return annotation, so the
    schema and the wire (a dict annotation = structuredContent) are the tool's own."""

    def __init__(self, mcp: Any, session: Session) -> None:
        self._mcp = mcp
        self._session = session

    def tool(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
        register = self._mcp.tool(*args, **kwargs)

        def decorate(fn: Callable[..., Any]) -> Any:
            @functools.wraps(fn)
            def call(*a: Any, **kw: Any) -> Any:
                try:
                    return fn(*a, **kw)
                finally:
                    self._session.report()

            return register(call)

        return decorate


def build_server(spec: BridgeSpec) -> Any:
    """The FastMCP server of one episode: every tool the spec names, registered in order on
    one shared Session. A tool the spec carries no description for is refused here."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mip-env")
    session = Session(spec)
    descriptions = spec["descriptions"]
    for ref in spec["tools"]:
        module = tool_module(ref)
        if module.NAME not in descriptions:
            raise SystemExit(
                f"{SPEC_ENV}: no description for tool {module.NAME!r} ({ref}) — "
                f"ToolSet.render_env renders one for every tool of the set"
            )
        module.register(_Reporting(mcp, session), session, descriptions[module.NAME])
    return mcp
