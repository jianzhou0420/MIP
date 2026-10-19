"""The harness protocol — what crosses the process boundary between an adapter and a harness
that MIP launches itself (mini).

The adapter writes one ``EpisodeSpec`` as JSON and starts the harness with its path; the
harness writes one JSON object per stdout line: ``event`` lines carry the bus vocabulary
(``kind`` + ``data``, exactly what ``sink.emit`` takes), ``raw`` lines a harness-native record,
and the last line is ``outcome`` (the ``SessionOutcome`` fields) or ``fatal`` (the session never
started). The adapter side is ``mip.harnesses.process.ProtocolAdapter``; the harness side is
``read_spec`` and ``Channel``, the only two names a harness process imports from here.

The model's API key never touches the spec file: the adapter passes it in ``API_KEY_ENV`` and
``read_spec`` puts it back into ``model.kwargs``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

#: the protocol's version, written into every spec
CONTRACT = 1
#: the environment variable that carries the model's API key past the spec file
API_KEY_ENV = "MIP_MODEL_API_KEY"


class ToolServer(TypedDict):
    """The experiment's MCP bridge as a stdio server: how to start it."""

    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str


class ModelCall(TypedDict):
    """The model as the harness calls it: the harness-facing id and the call kwargs
    (api_base, reasoning_effort, sampling params …), the API key excepted."""

    id: str
    kwargs: dict[str, Any]


class EpisodeSpec(TypedDict):
    """Everything one episode hands a harness process."""

    contract: int
    index: int
    briefing: str
    first_prompt: str
    instruction: str
    max_turns: int
    timeout_s: int
    workdir: str
    raw_dir: str
    server_url: str
    model: ModelCall
    tools: ToolServer
    #: the harness's own knobs; open keys, each harness reads its own
    harness: dict[str, Any]


LineType = Literal["event", "raw", "outcome", "fatal"]


def write_spec(spec: EpisodeSpec, path: Path) -> Path:
    """The adapter's half: the spec as JSON at ``path`` (the API key must already be out)."""
    if "api_key" in spec["model"]["kwargs"]:
        raise ValueError("the API key rides API_KEY_ENV, never the spec file")
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    return path


def read_spec(path: str | Path) -> EpisodeSpec:
    """The harness's half: the spec, with the API key from ``API_KEY_ENV`` back in
    ``model.kwargs`` (and out of the environment). Refuses a spec of another protocol version."""
    spec: EpisodeSpec = json.loads(Path(path).read_text())
    if spec.get("contract") != CONTRACT:
        raise ValueError(f"harness protocol {spec.get('contract')!r}, this side speaks {CONTRACT}")
    key = os.environ.pop(API_KEY_ENV, None)  # the bridge this process spawns never sees it
    if key:
        spec["model"]["kwargs"]["api_key"] = key
    return spec


class Channel:
    """The harness's stdout as protocol lines. Opening it moves fd 1 onto stderr, so a
    library that prints (litellm, mini's banner) lands in the stderr log, never in the
    stream the adapter parses."""

    def __init__(self) -> None:
        sys.stdout.flush()
        self._out = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        self._line({"type": "event", "kind": kind, "data": data})

    def raw(self, record: Any) -> None:
        self._line({"type": "raw", "record": record})

    def outcome(
        self,
        *,
        usage: dict[str, Any] | None,
        cost_usd: float | None,
        turns: int | None,
        error: str | None,
        extra: dict[str, Any],
    ) -> None:
        self._line(
            {
                "type": "outcome",
                "usage": usage,
                "cost_usd": cost_usd,
                "turns": turns,
                "error": error,
                "extra": extra,
            }
        )

    def fatal(self, error: str) -> None:
        self._line({"type": "fatal", "error": error})

    def _line(self, obj: dict[str, Any]) -> None:
        try:
            text = json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError):
            from mip.core.panel import json_safe

            text = json.dumps(json_safe(obj), ensure_ascii=False)
        self._out.write(text + "\n")
