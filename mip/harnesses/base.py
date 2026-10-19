"""The harness contract: what an adapter offers so the runner can drive a seat.

An adapter wraps one agent stack — Claude Agent SDK, Codex CLI, mini-swe-agent,
or the scripted fake — and runs ONE clean session per episode. The runner calls
four things: ``prepare`` / ``finalize`` once per run, ``describe`` / ``run`` once
per episode. The first three are templates here, so what every adapter would
otherwise repeat is written once: the settings box, the ``Harness`` record the
inherent facts start from, the gateway of a fake or cross-API seat, the refusal
of sampling params, the version learned at prepare, and the effort rule. A
subclass writes ``_options`` — its one translation of an ``EpisodeContext`` into
what its harness is given, which ``describe`` records — and ``run``; it adds
``_prepare`` / ``_finalize`` / ``_version`` when it has run-level facts to learn.

The adapters themselves are one module per harness in the root ``mip/harnesses/`` package
(``claude_sdk``, ``codex_cli``, ``mini_swe``); ``mip.harnesses.fake`` wraps no model.

Effort, the one rule: the seat word (``ctx.effort``) is sent when
it is a level; for ``default`` the harness constant ``settings["effort"]`` is
sent when there is one, else nothing — the product's own default applies
(``Harness.effort_default`` says what that is for each harness) and the
record says so (``inherent["effort_policy"]``, ``session_inputs.effort_sent``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from mip.core.episode import EpisodeContext, EventSink, ModelSpec, SessionOutcome
from mip.core.panel import json_safe

#: seat words that leave the effort to the harness (ModelSpec already turns "" into "default")
EFFORT_UNSET = (None, "", "default")


@dataclass(frozen=True)
class Harness:
    """The agent stack an adapter wraps, as the facts that hold before any run.

    Each is a whole scaffolding around a model — the loop, the tool protocol, how
    the briefing is delivered, what it can promise about turns and context, what
    it does with an effort it is not given, and what keeps the machine's own
    product configuration out of a session. An adapter carries exactly one; the
    recorded ``inherent`` dict starts from ``as_record()`` and ``prepare`` adds
    what a run learns (version, gateway, CLI pin, serving).
    """

    product: str  # Claude Agent SDK | Codex CLI | mini-swe-agent | scripted fake
    wire: str | None  # anthropic-messages | openai-chat (a gateway seat routes through mip.core.api);
    # litellm (mini speaks it itself: only the scripted fake endpoint applies); None = no gateway
    auth: str  # whose account pays, and how the credential reaches the harness
    thinking: str  # what of the model's reasoning is configurable, and what is visible
    turn_cap: str  # whether a turn limit exists and how hard it is
    system_prompt: str  # how the briefing reaches the model
    tool_surface: str  # how the bridge's tools are exposed to the model
    vision_context: str  # what happens to images across turns
    context_control: bool  # whether the adapter edits the resident transcript
    context_label: str  # the label every manifest of this harness carries
    effort_default: str  # what the product does when no effort is sent
    isolation: str  # what keeps the machine's product config out of a session
    version: str | None = None  # learned at prepare; None until then

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


class HarnessAdapter(ABC):
    """One agent stack behind the runner's four calls.

    A subclass writes ``_options`` and ``run``; it adds ``_prepare`` for the facts
    a run learns about it (a CLI pin, a serving stack, an auth decision),
    ``_finalize`` for a run-level audit, ``_version`` when the harness can tell its
    version. The base holds what every adapter carries — ``settings`` (the
    constructor kwargs as given: the harness file beside its ``_target_``), the
    ``Harness`` it wraps, the recorded ``inherent`` facts, the gateway of the
    current run — and the lines every adapter would otherwise repeat.
    """

    #: the seat word (``harness=<name>``)
    name: str
    #: whether ``model.<name>.<seat>.params`` (sampling params) reach the model; the
    #: others refuse them at prepare rather than ignore them
    accepts_params: bool = False

    def __init__(self, settings: dict[str, Any], harness: Harness) -> None:
        self.settings: dict[str, Any] = dict(settings)
        self.harness = harness
        self.inherent: dict[str, Any] = harness.as_record()
        self._gateway: Any = None  # the loopback endpoint of this run, or None for the vendor's own

    # ── once per run ────────────────────────────────────────────────

    def prepare(self, spec: ModelSpec) -> None:
        """Once per run, BEFORE any episode: refuse what this harness cannot honour,
        open the gateway of a fake / cross-API seat, let the adapter learn its
        run-level facts (``_prepare``), then its version. Raise to abort the cell:
        a misconfigured seat that silently degrades is worse than no run."""
        if spec.extra.get("params") and not self.accepts_params:
            raise RuntimeError(
                f"{self.name} takes no sampling params (model.<name>.<seat>.params) — "
                "this harness exposes none; drop them or use harness=mini"
            )
        self._gateway = self._open_gateway(spec)
        if self._gateway is not None:
            desc = self._gateway.describe()
            self.inherent["api_gateway"] = desc
            self.inherent["auth"] = desc["auth"]
            print(f"[{self.name}] api gateway :{self._gateway.port} — {desc['backend']}")
        self.inherent.update(self._prepare(spec))
        self.harness = replace(self.harness, version=self._version())
        self.inherent["version"] = self.harness.version
        self.inherent["effort_policy"] = self._effort_policy()

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        """Once per run, AFTER the last episode: the gateway goes down, then whatever
        audit the adapter alone can produce (``_finalize``), merged into
        summary.run_stats."""
        if self._gateway is not None:
            self._gateway.stop()
            self._gateway = None
        return self._finalize(run_dir)

    def _open_gateway(self, spec: ModelSpec) -> Any:
        """The loopback endpoint this run is pointed at, by the harness's wire: the
        scripted fake endpoint of an ``api=fake`` run, or the litellm gateway of a
        cross-API seat; None for a native seat or a harness without a wire."""
        wire = self.harness.wire
        if wire is None:
            return None
        from mip.core.api import (  # lazy: mip.core.api brings the serving stacks
            fake_gateway_for,
            gateway_for,
        )

        if wire == "litellm":
            return fake_gateway_for(spec, tag=self.name)
        return gateway_for(spec, wire=wire, tag=self.name)

    def _prepare(self, spec: ModelSpec) -> dict[str, Any]:
        """Run-level facts this adapter learns at prepare, merged into ``inherent``;
        the gateway (if any) is already open."""
        return {}

    def _finalize(self, run_dir: Path) -> dict[str, Any]:
        """Serving-side accounting after a run, where the adapter has any; ``{}`` otherwise."""
        return {}

    def _version(self) -> str | None:
        """The wrapped harness's version, learned at prepare; None when it cannot be told."""
        return None

    # ── once per episode ────────────────────────────────────────────

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        """The harness block of the session_inputs record: the seat's effort word,
        the effort actually sent (None = key omitted), and ``options`` — the
        ``_options`` translation, so the record cannot drift from the request.
        A subclass adds what else it has to say."""
        return {
            "effort": ctx.effort,
            "effort_sent": self._effort(ctx),
            "options": json_safe(self._options(ctx)),
        }

    @abstractmethod
    def _options(self, ctx: EpisodeContext) -> Any:
        """What this episode hands the harness, built from the context — the SDK's
        options object, a CLI's argv, a library's kwargs. ``run`` sends it and
        ``describe`` records it (json-safe): one translation."""

    @abstractmethod
    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        """Run ONE clean session; emit events through ``sink.emit`` only and hand
        every harness-native record to ``sink.raw`` (mip.core.bus.EventBus)."""

    # ── what every adapter would otherwise repeat ──

    def _effort(self, ctx: EpisodeContext) -> str | None:
        """The effort to send: the seat word when it is a level; else the harness
        constant ``settings["effort"]``; else None — nothing is sent."""
        if ctx.effort not in EFFORT_UNSET:
            return str(ctx.effort)
        constant = self.settings.get("effort")
        return None if constant in EFFORT_UNSET else str(constant)

    def _effort_policy(self) -> str:
        constant = self.settings.get("effort")
        if constant in EFFORT_UNSET:
            return (
                "a seat effort is sent as given; for `default` nothing is sent — "
                f"{self.harness.effort_default}"
            )
        return (
            f"a seat effort is sent as given; for `default` the harness constant "
            f"{constant!r} is sent"
        )

    def _setting(self, ctx: EpisodeContext, key: str) -> Any:
        """A per-run harness knob: the run cfg's extra by name, else the harness file's setting."""
        return ctx.extra.get(key, self.settings[key])

    def _write_manifest(self, ctx: EpisodeContext, **fields: Any) -> None:
        """The provider-payload manifest of a closed harness: what the
        session accumulated, counted from the record stream the adapter sees, under
        the harness's own context label. Telemetry only — a write failure never
        touches the SessionOutcome."""
        try:
            manifest = {
                "label": self.harness.context_label,
                "context_control": self.harness.context_control,
                "episode": ctx.index,
                **fields,
            }
            (ctx.raw_dir / f"context_manifest_{ctx.index}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1)
            )
        except Exception:
            pass


# ── module helpers the CLI adapters share ──


def resolve_cli(cli_path: str | None, *, fallback: str | None, tag: str) -> str | None:
    """harness.cli_path as an absolute path. Null → ``fallback``: a binary name to
    find on PATH (its absolute path, or an error), or None for the harness's own
    bundled CLI. A path that is not an executable file raises: a cell that fell
    back to another CLI unnoticed would be a different cell."""
    if not cli_path:
        if fallback is None:
            return None
        found = shutil.which(fallback)
        if not found:
            raise RuntimeError(f"[{tag}] no `{fallback}` on PATH and harness.cli_path is null")
        return os.path.abspath(found)
    path = os.path.abspath(os.path.expanduser(str(cli_path)))
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        raise RuntimeError(f"[{tag}] harness.cli_path is not an executable file: {path}")
    return path


def native_binary(cli: str) -> Path | None:
    """The native binary behind an npm shim (``vendor/<triple>/bin/<name>`` of the
    package or its platform sub-package), or the CLI itself when it is already native."""
    entry = Path(cli).resolve()
    if entry.suffix != ".js":
        return entry
    name = entry.stem
    pkg = entry.parent.parent  # .../<scope>/<package>
    hits = sorted(pkg.glob(f"node_modules/@*/{name}-*/vendor/*/bin/{name}"))
    hits += sorted(pkg.glob(f"vendor/*/bin/{name}"))
    return hits[0] if hits else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cli_pin(
    cli: str, *, env: dict[str, str] | None = None, timeout_s: float = 30.0
) -> dict[str, Any]:
    """The CLI a run is pinned to, as a record: its path, the native binary behind
    it, that binary's sha256, and what ``<cli> --version`` prints. A CLI that
    cannot report its version aborts the run (fail closed)."""
    native = native_binary(cli)
    version = subprocess.run(
        [cli, "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_s,
        env=env,
    ).stdout.strip()
    return {
        "path": cli,
        "native": str(native) if native else None,
        "sha256": sha256_of(native) if native else None,
        "version": version,
    }


def tool_result_texts(content: Any) -> list[str]:
    """The text parts of a tool result's content in either wire shape, an image
    replaced by its marker (the frames live in live_*/)."""
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    texts.append("<image elided>")
    return texts
