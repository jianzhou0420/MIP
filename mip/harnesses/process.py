"""The harness as a process: every adapter whose harness runs as its own process, one per episode.

``ProcessAdapter.run`` is the one session loop they share — start the command, read its stdout
one JSON object per line, hand each to the subclass, reap the process group however the session
ends — so a subclass writes only what differs: the command (``_argv``), its environment
(``_env``), what a line means (``_on_line``) and what the session came to (``_outcome``). The
Codex CLI is one (its own ``--json`` events); ``ProtocolAdapter`` is the other kind, a harness
MIP launches itself and that speaks ``mip.harnesses.protocol`` (mini).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any

from mip.core.episode import EpisodeContext, EventSink, SessionOutcome
from mip.harnesses.base import HarnessAdapter
from mip.harnesses.protocol import API_KEY_ENV, CONTRACT, EpisodeSpec, write_spec
from mip.core.panel import json_safe

#: the stdout line limit when the settings name none: look_around returns four images in one line
MAX_LINE = 32 * 1024 * 1024
#: how much of the stderr log a crashed session's error carries
STDERR_TAIL = 2000


class HarnessProcessError(RuntimeError):
    """The harness process ended without telling the adapter how the session went."""


class ProcessAdapter(HarnessAdapter):
    """An adapter whose harness is a process started once per episode.

    A subclass writes ``_argv``, ``_env``, ``_on_line`` and ``_outcome``; ``_begin`` opens the
    per-episode state the other three share. ``record_raw`` says whether every line also goes
    to ``sink.raw`` as the harness-native record."""

    #: every stdout line is also a raw record (raw/episode_<i>.jsonl)
    record_raw: bool = True

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        state = self._begin(ctx)
        argv = self._argv(ctx, state)
        stderr_path = ctx.raw_dir / f"episode_{ctx.index}.stderr.log"
        with stderr_path.open("wb") as stderr_file:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_file,
                cwd=str(ctx.workdir),
                env=self._env(ctx, state),
                start_new_session=True,  # own process group: the kill reaps the MCP bridge too
                limit=int(self.settings.get("max_buffer_size") or MAX_LINE),
            )
            try:
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        sink.emit(
                            "driver_error",
                            {"error": {"nonjson_stdout": line[:300].decode(errors="replace")}},
                        )
                        continue
                    if self.record_raw:
                        sink.raw({"event": json_safe(obj)})
                    self._on_line(obj, state, sink)
                exit_code = await proc.wait()
            finally:
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
        return self._outcome(ctx, state, exit_code)

    def _begin(self, ctx: EpisodeContext) -> dict[str, Any]:
        """The per-episode state ``_on_line`` and ``_outcome`` share; empty by default."""
        return {}

    @abstractmethod
    def _argv(self, ctx: EpisodeContext, state: dict[str, Any]) -> list[str]:
        """The command line of one episode's session."""

    @abstractmethod
    def _env(self, ctx: EpisodeContext, state: dict[str, Any]) -> dict[str, str]:
        """The whole environment the session process gets."""

    @abstractmethod
    def _on_line(self, obj: dict[str, Any], state: dict[str, Any], sink: EventSink) -> None:
        """One parsed stdout line: emit what it means, keep what ``_outcome`` needs."""

    @abstractmethod
    def _outcome(
        self, ctx: EpisodeContext, state: dict[str, Any], exit_code: int
    ) -> SessionOutcome:
        """What the session came to, once the process has exited."""


class ProtocolAdapter(ProcessAdapter):
    """A harness MIP launches itself: ``sys.executable <entry> --episode <spec>``, speaking
    ``mip.harnesses.protocol``. A subclass names the ``entry`` script and writes ``_spec``
    (what this harness is given, built from its ``_options``); the lines, the outcome and the
    key handling are the protocol's."""

    #: the harness's process entry, a script path
    entry: Path
    #: protocol lines are the bus vocabulary already; a raw line is the harness's own record
    record_raw = False

    @abstractmethod
    def _spec(self, ctx: EpisodeContext) -> EpisodeSpec:
        """The episode's spec. ``model.kwargs`` may hold ``api_key``: ``_argv`` moves it to
        the environment before the spec is written."""

    def _base_spec(self, ctx: EpisodeContext) -> dict[str, Any]:
        """The spec fields every protocol harness gets the same way, ``model`` and ``harness``
        left to the subclass."""
        return {
            "contract": CONTRACT,
            "index": ctx.index,
            "briefing": ctx.briefing,
            "first_prompt": ctx.first_prompt,
            "instruction": ctx.instruction,
            "max_turns": int(ctx.max_turns),
            "timeout_s": int(ctx.episode_timeout),
            "workdir": str(ctx.workdir),
            "raw_dir": str(ctx.raw_dir),
            "server_url": ctx.server_url,
            "tools": {
                "command": sys.executable,
                "args": [str(ctx.bridge_path)],
                "env": ctx.bridge_env(),
                "cwd": str(Path(ctx.bridge_path).parent),
            },
        }

    def _begin(self, ctx: EpisodeContext) -> dict[str, Any]:
        spec = self._spec(ctx)
        key = spec["model"]["kwargs"].pop("api_key", None)
        path = write_spec(spec, ctx.raw_dir / f"episode_{ctx.index}.spec.json")
        return {"spec_path": path, "api_key": key, "outcome": None, "fatal": None}

    def _argv(self, ctx: EpisodeContext, state: dict[str, Any]) -> list[str]:
        return [sys.executable, str(self.entry), "--episode", str(state["spec_path"])]

    def _env(self, ctx: EpisodeContext, state: dict[str, Any]) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV}
        if state["api_key"]:
            env[API_KEY_ENV] = str(state["api_key"])
        return env

    def _on_line(self, obj: dict[str, Any], state: dict[str, Any], sink: EventSink) -> None:
        kind = obj.get("type")
        if kind == "event":
            sink.emit(str(obj.get("kind")), obj.get("data") or {})
        elif kind == "raw":
            sink.raw(obj.get("record"))
        elif kind == "outcome":
            state["outcome"] = obj
        elif kind == "fatal":
            state["fatal"] = str(obj.get("error"))
        else:
            sink.emit("driver_error", {"error": {"unknown_protocol_line": json_safe(obj)}})

    def _outcome(
        self, ctx: EpisodeContext, state: dict[str, Any], exit_code: int
    ) -> SessionOutcome:
        if state["fatal"] is not None:
            raise HarnessProcessError(f"[{self.name}] {state['fatal']}")
        out = state["outcome"]
        if out is None:
            tail = (ctx.raw_dir / f"episode_{ctx.index}.stderr.log").read_text(errors="replace")
            raise HarnessProcessError(
                f"[{self.name}] harness exited {exit_code} without an outcome: "
                f"{tail[-STDERR_TAIL:]}"
            )
        return SessionOutcome(
            usage=out.get("usage"),
            cost_usd=out.get("cost_usd"),
            turns=out.get("turns"),
            error=out.get("error"),
            extra=out.get("extra") or {},
        )
