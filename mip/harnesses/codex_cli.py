"""OpenAI Codex CLI adapter: ``codex exec --json``, one process per episode.

The contract and what every adapter shares are in mip.harnesses.base, the one-process-per-episode
session loop in mip.harnesses.process (this adapter writes its argv, env and event translation).
codex keeps its built-in system prompt (that closed scaffolding is the thing under test); the
briefing rides as the one user prompt. Auth is the logged-in ChatGPT subscription. Codex-specific
wiring:

- MCP tool calls are approval-gated and exec mode auto-cancels them; the CLI accepts only
  prompt|approve for default_tools_approval_mode ("auto" is silently invalid).
- no CLI-level turn cap exists: ctx.max_turns feeds the bridge broadcast / STOP gate only
  (recorded difference; hard caps = step budget + timeout).
- reasoning is usage-counted but unreadable (summary=[] + encrypted content).
- codex runs in "code mode": MCP tools are not sent to the model as function tools; they are
  deferred, listed in the JS ``exec`` custom tool's ALL_TOOLS and called from JS
  (``await tools.mcp__env__step({...})``). Direct function calls cannot be restored
  (``features.code_mode_host = false`` fails closed: no exec host, no tools). So this seat's
  tool surface differs from the SDK and mini seats', recorded as harness_inherent.tool_surface;
  the fake endpoint (mip/core/api/fake_wire.py) drives exec the same way.
- isolation from the machine: codex reads its whole state from CODEX_HOME (config.toml, where an
  operator's model_reasoning_effort silently replaces the product default; skills, plugins, the
  memories store, past sessions). Every episode therefore runs in a PRIVATE home under
  <run>/codex_home/episode_<i>/ that holds one symlink, auth.json -> the real login file (codex
  rewrites it in place on token refresh, so the link keeps the real login current), nothing
  else; codex refreshes the model catalog into it (models_cache.json = the product defaults at
  run time) and writes its session rollouts there. The subprocess env is scrubbed of CODEX_* /
  OPENAI_* / CLAUDE* / ANTHROPIC*, and the CLI is a pinned path with its version and the native
  binary's sha256 recorded (harness.cli_path; null = the `codex` on PATH).
- effort: the seat word `default` sends NO model_reasoning_effort; with no config.toml in the
  private home codex fills the request from the product catalog's default_reasoning_level for
  the model (the catalog it fetches into the private home at start, cached as
  models_cache.json). The adapter reads that catalog after the episode and records the level
  that applied and the catalog's fetched_at (agent.effort_applied / catalog_fetched_at).
  harness.effort = null selects this (the one effort rule, mip.harnesses.base); a level there is
  sent as a constant. Any other seat word (low, high, …) is sent explicitly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from mip.core.episode import EpisodeContext, EventSink, ModelSpec, SessionOutcome
from mip.harnesses.base import Harness, cli_pin, resolve_cli, tool_result_texts
from mip.harnesses.process import ProcessAdapter
from mip.core.panel import json_safe


def _toml_str(value: str) -> str:
    # TOML basic strings cannot carry raw newlines/tabs — escape them or a
    # crafted value splits the -c override into a second TOML statement
    return (
        '"'
        + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        + '"'
    )


# env prefixes that never reach the codex subprocess: codex's own (CODEX_HOME
# is set by the adapter), OpenAI keys (subscription auth only), and the host
# Claude Code session's variables (CLAUDE_EFFORT, CLAUDECODE, ANTHROPIC_*)
HOST_ENV_PREFIXES = ("CODEX_", "OPENAI_", "CLAUDE", "ANTHROPIC")


def _real_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _private_codex_home(home: Path) -> Path:
    """Create an empty codex home holding only auth.json -> the real login
    file. A missing real login is not an error here (fake-endpoint runs need
    none); a missing login surfaces as codex's own auth failure."""
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o700)
    real = _real_codex_home() / "auth.json"
    link = home / "auth.json"
    if real.is_file() and not link.exists():
        link.symlink_to(real)
    return home


def _codex_env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(HOST_ENV_PREFIXES)}
    env["CODEX_HOME"] = str(home)
    return env


class CodexCliAdapter(ProcessAdapter):
    name = "codex"

    def __init__(
        self,
        *,
        effort: str | None = None,
        model_reasoning_summary: str = "detailed",
        tools_approval_mode: str = "approve",
        project_doc_max_bytes: int = 0,
        sandbox: str = "read-only",
        probe_timeout_s: float = 45.0,
        max_buffer_size: int = 32 * 1024 * 1024,
        cli_path: str | None = None,
    ) -> None:
        # the harness's own constants (constructor kwargs beside its _target_);
        # `effort` is what the seat word `default` sends: null = nothing (the
        # key omitted, codex fills the product catalog's default for the
        # model), a level = sent as a constant — a run's own effort comes in
        # as ctx.effort
        scrub = " ".join(f"{p}*" for p in HOST_ENV_PREFIXES)
        super().__init__(
            settings={
                "effort": effort,
                "model_reasoning_summary": model_reasoning_summary,
                "tools_approval_mode": tools_approval_mode,
                "project_doc_max_bytes": project_doc_max_bytes,
                "sandbox": sandbox,
                "probe_timeout_s": probe_timeout_s,
                "max_buffer_size": max_buffer_size,
                "cli_path": cli_path,
            },
            harness=Harness(
                product="Codex CLI",
                wire="openai-chat",
                auth="ChatGPT subscription (codex login)",
                thinking="reasoning tokens counted, content unreadable",
                turn_cap="broadcast only (no codex-side hard cap)",
                system_prompt="codex builtin; the briefing heads the one user prompt",
                # code mode: MCP tools reach the model only through its JS exec
                # tool (deferred, ALL_TOOLS), not the direct function calls of
                # the SDK and mini seats. See the module docstring.
                tool_surface="codex code mode: MCP tools deferred, called from the JS exec tool",
                # codex CLI owns its own context/vision management; its image
                # retention policy is not audited here.
                vision_context="codex-managed (unaudited)",
                # codex owns its own history (retention count, role structure),
                # which this repo does not audit, so runs carry the unaudited label.
                context_control=False,
                context_label="codex-managed-unaudited",
                effort_default=(
                    "the product catalog's default_reasoning_level for the model "
                    "(model_reasoning_effort omitted; codex fills it from the catalog it fetched "
                    "into the private home — recorded per episode as agent.effort_applied)"
                ),
                # nothing of the operator's ~/.codex reaches a session
                isolation=(
                    "private CODEX_HOME per episode (<run>/codex_home/episode_<i>/): auth.json "
                    "linked to the real login, no config.toml / skills / plugins / memories / "
                    f"sessions from the machine; subprocess env without {scrub}"
                ),
            ),
        )
        self._cli: str = "codex"  # resolved in _prepare()
        self._cli_version: str | None = None

    def _prepare(self, spec: ModelSpec) -> dict[str, Any]:
        # the CLI this run is pinned to: path, native binary, sha256, version
        # (probed in a temp private home with the scrubbed env, like an episode)
        self._cli = resolve_cli(self.settings["cli_path"], fallback="codex", tag="codex")
        pin = cli_pin(
            self._cli,
            env=_codex_env(_private_codex_home(Path(tempfile.mkdtemp(prefix="codex-home-")))),
            timeout_s=float(self.settings["probe_timeout_s"]),
        )
        self._cli_version = pin["version"]
        print(f"[codex] cli -> {self._cli} ({pin['version']}, native sha256 {pin['sha256']})")
        return {"cli": pin}

    def _version(self) -> str | None:
        return self._cli_version

    def _options(self, ctx: EpisodeContext) -> list[str]:
        """The ``codex exec`` command line of one episode (the prompt is appended
        by ``run``): model, effort, the bridge as an MCP server, the gateway."""
        bridge_env = ctx.bridge_env()
        s = self.settings
        env_table = ", ".join(f"{k} = {_toml_str(v)}" for k, v in bridge_env.items())
        return [
            self._cli,
            "exec",
            "--json",
            "--skip-git-repo-check",
            # the sandbox policy for model-run shell commands (read-only |
            # workspace-write | danger-full-access); read-only = exec's default
            "--sandbox",
            str(s["sandbox"]),
            # blank model = the CLI's own default; `-c model = ""` would 400
            *(["-c", f"model = {_toml_str(ctx.model)}"] if ctx.model else []),
            # None = omitted: codex fills the product catalog's default (base._effort)
            *(
                ["-c", f"model_reasoning_effort = {_toml_str(effort)}"]
                if (effort := self._effort(ctx)) is not None
                else []
            ),
            # Display-layer knob: surfaces reasoning summaries if any exist.
            "-c",
            f"model_reasoning_summary = {_toml_str(s['model_reasoning_summary'])}",
            "-c",
            f"mcp_servers.env.command = {_toml_str(sys.executable)}",
            "-c",
            f"mcp_servers.env.args = [{_toml_str(str(ctx.bridge_path))}]",
            "-c",
            f"mcp_servers.env.env = {{ {env_table} }}",
            "-c",
            f"mcp_servers.env.default_tools_approval_mode = {_toml_str(s['tools_approval_mode'])}",
            # No AGENTS.md injection (the SDK seat's setting_sources=[] analog).
            "-c",
            f"project_doc_max_bytes = {int(s['project_doc_max_bytes'])}",
            # a gateway seat: route this run through the owned loopback endpoint
            *(self._gateway.codex_args() if self._gateway is not None else []),
        ]

    @staticmethod
    def _catalog_default(home: Path, model: str) -> tuple[str | None, str | None, str]:
        """(default_reasoning_level for `model`, fetched_at, status) from the
        catalog codex fetched into `home`; status says when it is missing —
        a custom provider (fake endpoint, gateway) never fetches one and
        codex then falls back to the list bundled in the binary."""
        cache = home / "models_cache.json"
        if not cache.is_file():
            return (
                None,
                None,
                "missing (custom provider or fetch failure: codex used its bundled list)",
            )
        try:
            data = json.loads(cache.read_text())
        except ValueError:
            return None, None, "unreadable"
        for entry in data.get("models") or []:
            if entry.get("slug") == model:
                return entry.get("default_reasoning_level"), data.get("fetched_at"), "fetched"
        return None, data.get("fetched_at"), "fetched (model not listed)"

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {
            **super().describe(ctx),
            "sandbox": self.settings["sandbox"],
            "system_prompt_note": "<codex builtin>",
        }

    # ── one session: mip.harnesses.process.ProcessAdapter runs it ──

    def _begin(self, ctx: EpisodeContext) -> dict[str, Any]:
        run_dir = ctx.run_dir or ctx.raw_dir.parent
        return {
            "home": _private_codex_home(run_dir / "codex_home" / f"episode_{ctx.index}"),
            "model": ctx.model,
            "usage_totals": {},
            "thread_id": None,
            "n_turns": 0,
            "n_tool_results": 0,
            "n_images": 0,
            "n_state_blocks": 0,
        }

    def _argv(self, ctx: EpisodeContext, state: dict[str, Any]) -> list[str]:
        prompt = ctx.briefing + "\n\n" + ctx.first_prompt
        # `--` ends option parsing so a prompt starting with "-" cannot be
        # read as a flag
        return [*self._options(ctx), "--", prompt]

    def _env(self, ctx: EpisodeContext, state: dict[str, Any]) -> dict[str, str]:
        return _codex_env(state["home"])

    def _on_line(self, event: dict[str, Any], state: dict[str, Any], sink: EventSink) -> None:
        kind = event.get("type")
        if kind == "thread.started":
            state["thread_id"] = event.get("thread_id")
            sink.emit("system_init", {"thread_id": state["thread_id"], "model": state["model"]})
            return
        if kind == "turn.completed":
            state["n_turns"] += 1
            totals = state["usage_totals"]
            for key, value in (event.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + int(value)
            return
        if kind in ("turn.failed", "error"):
            sink.emit("driver_error", {"error": json_safe(event)})
            return
        item = event.get("item") or {}
        item_type = item.get("type")
        if kind == "item.started" and item_type == "mcp_tool_call":
            sink.emit(
                "tool_use",
                {
                    "id": item.get("id"),
                    "name": f"mcp__{item.get('server')}__{item.get('tool')}",
                    "input": item.get("arguments"),
                },
            )
            return
        if kind == "item.completed":
            if item_type == "mcp_tool_call":
                result = item.get("result")
                texts = tool_result_texts(
                    result.get("content") if isinstance(result, dict) else None
                )
                if item.get("error"):
                    texts.append(json.dumps({"error": json_safe(item["error"])}))
                state["n_tool_results"] += 1
                state["n_images"] += sum(1 for t in texts if t == "<image elided>")
                state["n_state_blocks"] += sum(1 for t in texts if "[STATE" in t)
                sink.emit("tool_result", {"tool_use_id": item.get("id"), "texts": texts})
            elif item_type == "agent_message":
                sink.emit("assistant_text", {"text": item.get("text", "")})
            elif item_type == "reasoning":
                text = item.get("text") or ""
                sink.emit("thinking", {"chars": len(text), "text": text})
            elif item_type == "command_execution":
                # codex's own shell tool — can't be unmounted; sandbox is
                # read-only and every use is on the record.
                sink.emit(
                    "tool_use",
                    {
                        "id": item.get("id"),
                        "name": "shell",
                        "input": {"command": item.get("command")},
                    },
                )
                sink.emit(
                    "tool_result",
                    {
                        "tool_use_id": item.get("id"),
                        "texts": [str(item.get("aggregated_output", ""))[:4000]],
                    },
                )
            else:
                sink.emit("driver_error", {"error": {"unhandled_item": json_safe(item)}})

    def _outcome(
        self, ctx: EpisodeContext, state: dict[str, Any], exit_code: int
    ) -> SessionOutcome:
        usage_totals = state["usage_totals"]
        # mirrors claude_sdk's manifest, with one difference: every count comes
        # from the EVENT STREAM, not the provider payload, and codex "turns"
        # are its own units, not LLM calls.
        self._write_manifest(
            ctx,
            num_turns=state["n_turns"],  # codex turn.completed events
            usage=usage_totals or None,  # summed client-side; unaudited
            tool_results=state["n_tool_results"],
            images=state["n_images"],
            state_blocks=state["n_state_blocks"],
            note=(
                "counts from the codex event stream, not the "
                "resident provider payload — codex owns its own "
                "transcript and no per-request audit exists here"
            ),
        )
        sent = self._effort(ctx)
        level, fetched_at, status = self._catalog_default(state["home"], ctx.model)
        return SessionOutcome(
            usage=usage_totals,
            cost_usd=None,  # subscription auth — no per-token billing
            turns=None,  # codex "turns" ≠ LLM calls; bridge counts tool calls
            error=(f"codex exited {exit_code}" if exit_code else None),
            extra={
                "thread_id": state["thread_id"],
                "exit_code": exit_code,
                "effort_sent": sent,  # None = key omitted
                "effort_applied": sent if sent is not None else level,
                "catalog": status,
                "catalog_fetched_at": fetched_at,
            },
        )
