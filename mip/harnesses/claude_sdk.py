"""Claude Agent SDK adapter: Claude Code, run as a CLI subprocess under the SDK.

The contract and what every adapter shares are in mip.harnesses.base. Auth rides the
logged-in Claude subscription; a stray ANTHROPIC_API_KEY would silently switch billing to the
API in headless mode, so the CLI subprocess gets it blanked (an empty value is unset for the
CLI; the SDK merges ClaudeAgentOptions.env over the process environment, so the process itself
is never edited).

To bill a run through the API instead, export CODING_AGENT_ALLOW_API_KEY=1 in the shell that
launches runner.py: the key then stays in the CLI's environment (ClaudeAgentOptions has no
api_key field; it is purely env-var-driven). Prompt, tools and bridge stay identical; only
billing changes. A gateway seat (api=fake, a cross-API litellm route) reaches the CLI the same
way: ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN in that environment table.

The SDK runs the Claude Code CLI bundled inside its own package, whatever `claude` is on PATH.
``harness.cli_path`` names another CLI binary for a model the bundled one refuses (the refusal
arrives as an ordinary few-second episode with no steps); null keeps the bundled CLI.

The CLI adds text of its own to every session unless told not to: a per-turn "batching
reminder" for some models behind a remote feature flag, the project's auto-memory index in the
first user message, a <total_tokens> line after every tool-result batch, a silent-turn nudge,
and one extra request per session that asks the model to name the session. A fixed protocol
needs them off and the remote flags frozen, so _options() passes CLI_QUIET_ENV into the CLI
subprocess and the harness record's ``isolation`` names it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

from mip.core.episode import EpisodeContext, EventSink, ModelSpec, SessionOutcome
from mip.harnesses.base import (
    Harness,
    HarnessAdapter,
    cli_pin,
    resolve_cli,
    tool_result_texts,
)
from mip.core.panel import json_safe
from mip.core.retry import is_rate_limited


def classify_outcome(result_msg: Any) -> str | None:
    """Map an SDK ResultMessage to the driver's error contract.

    The Agent SDK sets is_error=True even on a clean subtype="success"
    result — the flag tracks the session, not the navigation outcome, so
    an episode that called stop and reached the goal still comes back
    is_error=True. is_error alone therefore over-flags. Score by the ENV terminal instead: "success" (normal
    return, whatever the nav result), "error_max_turns" (clean
    truncation, like mini's step_limit), and "error_max_budget_usd"
    (USD fuse tripped — same clean-truncation semantics) are scored
    outcomes — only a genuine execution error (error_during_execution,
    or a missing subtype) is a broken session that propagates as error.
    Keeping the fuse subtype OUT of this whitelist would make the driver
    retry the most expensive episodes — the opposite of a budget cap.
    """
    subtype = getattr(result_msg, "subtype", None)
    result_text = str(getattr(result_msg, "result", "") or "")
    if is_rate_limited(result_text):
        # subscription throttle returns subtype="success" is_error=True with a
        # "temporarily limiting requests" result — tag retryable so the driver
        # backs off and re-runs it, never scoring it as a navigation failure.
        return "rate_limited"
    if getattr(result_msg, "is_error", False) and subtype not in (
        "error_max_turns",
        "error_max_budget_usd",
        "success",
    ):
        return f"sdk result {subtype or 'is_error'}"
    return None


# Env overrides that switch off the CLI's own prompt injections (each checked with api=fake;
# an EMPTY value counts as unset, and TOTAL_TOKENS_REMINDER wants the word "off", not 0).
# The SDK merges this dict into the CLI subprocess environment.
CLI_QUIET_ENV: dict[str, str] = {
    "CLAUDE_CODE_TOASTY_THIMBLE": "0",  # per-turn batching reminder (gate tengu_toasty_thimble, Fable models)
    "CLAUDE_CODE_GENTLE_PARASOL": "0",  # its secondary reminder (gate tengu_gentle_parasol)
    "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",  # <total_tokens>N tokens left</total_tokens> after each tool batch
    "CLAUDE_CODE_SILENT_TURN_REMINDER": "0",  # the "hasn't heard from you in a while" nudge
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",  # the project's memory index pasted into the first user message
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",  # the per-session "name this session" side request
    "DISABLE_GROWTHBOOK": "1",  # no remote feature flags: the switches above must not flip under us
}


class ClaudeSdkAdapter(HarnessAdapter):
    name = "claude-sdk"

    def __init__(
        self,
        *,
        effort: str | None = None,
        permission_mode: str = "bypassPermissions",
        strict_mcp_config: bool = True,
        max_buffer_size: int = 32 * 1024 * 1024,
        bridge_connect_timeout_s: float = 30.0,
        betas: list[str] | None = None,
        think_budget: int | None = None,
        haiku_think_budget: int = 4000,
        cli_path: str | None = None,
    ) -> None:
        # the harness's own constants — mip/configs/harness/cc.yaml beside its
        # _target_; a run's cfg extra overrides the per-run ones by name
        # (think_budget / betas / bridge_connect_timeout_s).
        # `effort` is what the seat word `default` sends: null = nothing, the
        # CLI's own per-model table applies; a level is sent as a constant.
        quiet = ", ".join(f"{k}={v}" for k, v in CLI_QUIET_ENV.items())
        super().__init__(
            settings={
                "effort": effort,
                "permission_mode": permission_mode,
                "strict_mcp_config": strict_mcp_config,
                "max_buffer_size": max_buffer_size,
                "bridge_connect_timeout_s": bridge_connect_timeout_s,
                "betas": list(betas or []),
                "think_budget": think_budget,
                "haiku_think_budget": haiku_think_budget,
                "cli_path": resolve_cli(cli_path, fallback=None, tag="claudecode"),
            },
            harness=Harness(
                product="Claude Agent SDK",
                wire="anthropic-messages",
                auth="claude subscription",
                thinking="adaptive+summarized (4.6+/5, haiku-5.5) / enabled+budget (haiku-4.5)",
                turn_cap="hard (SDK max_turns)",
                system_prompt="ClaudeAgentOptions.system_prompt (a true system role)",
                tool_surface="the tool set's stdio bridge as MCP function tools (mcp__env__*); no built-in tools",
                # CC/Agent-SDK does NOT evict old images per-turn (verified against
                # Claude Code source: applyToolResultBudget passes image tool_results
                # through as-is; the time-based clear is off by default). Images are
                # re-sent in FULL every turn, same as mini image_window=0. The only
                # backstop is full compaction (auto at ~167k tokens / reactive on a
                # media-size error), which drops the whole pre-boundary history and
                # replaces images with '[image]'. So request size grows with
                # accumulated frames until a compaction resets it — NOT a recent
                # window.
                vision_context="full history re-sent each turn; compaction backstop ~167k tok (no per-turn eviction)",
                # The SDK cannot edit its resident transcript, so the harness has
                # no context control: the session keeps its full history, and the
                # record labels it so.
                context_control=False,
                context_label="full-history",
                effort_default="the CLI's own per-model table, sent as output_config.effort",
                isolation=(
                    f"CLI subprocess env: {quiet}; setting_sources=[] (no CLAUDE.md, no settings "
                    "from disk); tools=[] (no built-in tools); ANTHROPIC_API_KEY blanked unless "
                    "CODING_AGENT_ALLOW_API_KEY=1"
                ),
            ),
        )
        self._blank_api_key = True  # decided once per run in _prepare
        self._cli_version: str | None = None

    def _prepare(self, spec: ModelSpec) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        allow_api_key = os.environ.get("CODING_AGENT_ALLOW_API_KEY") == "1"
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._blank_api_key = not (allow_api_key and has_key)
        if allow_api_key and has_key:
            print(
                "[claudecode] CODING_AGENT_ALLOW_API_KEY=1 — keeping ANTHROPIC_API_KEY, sessions bill via API"
            )
            facts["auth"] = "provider API key (litellm-free direct billing)"
        elif allow_api_key:
            print(
                "[claudecode] CODING_AGENT_ALLOW_API_KEY=1 but ANTHROPIC_API_KEY is unset — "
                "falling back to subscription auth"
            )
        elif has_key:
            print(
                "[claudecode] ANTHROPIC_API_KEY is set — blanked for the CLI so sessions use "
                "subscription auth (export CODING_AGENT_ALLOW_API_KEY=1 to bill via API instead)"
            )
        import claude_agent_sdk
        from claude_agent_sdk._cli_version import __cli_version__

        facts["sdk_version"] = getattr(claude_agent_sdk, "__version__", "?")
        cli = self.settings["cli_path"]
        if cli:
            facts["cli"] = cli_pin(cli)
            print(f"[claudecode] cli_path -> {cli} ({facts['cli']['version']})")
        else:
            # null = the SDK's bundled CLI (module docstring)
            facts["cli"] = {
                "path": None,
                "native": None,
                "sha256": None,
                "version": __cli_version__,
            }
        self._cli_version = facts["cli"]["version"]
        return facts

    def _version(self) -> str | None:
        return self._cli_version

    def _cli_env(self) -> dict[str, str]:
        """The CLI subprocess's environment table (the SDK merges it over the
        process env): the injection switches, the blanked API key, and the
        gateway redirect of a gateway seat."""
        env = dict(CLI_QUIET_ENV)
        if self._blank_api_key:
            env["ANTHROPIC_API_KEY"] = ""
        if self._gateway is not None:
            env.update(self._gateway.anthropic_env())
        return env

    # Thinking config is model-family-specific. Claude 4.6+/5 models take
    # `adaptive`; pre-4.6 (haiku-4.5) need explicit `enabled` + budget_tokens
    # and think ONCE per turn (no interleaved thinking without the beta).
    def _thinking_config(self, ctx: EpisodeContext) -> dict[str, Any]:
        # An explicit think_budget (from the experiment file or a command-line
        # override) forces enabled thinking so reasoning blocks are
        # substantive rather than adaptive one-liners.
        budget = self._setting(ctx, "think_budget")
        if budget:
            return {"type": "enabled", "budget_tokens": int(budget), "display": "summarized"}
        if "haiku-4-5" in (ctx.model or ""):  # pre-4.6 Haiku only; haiku-5-5 is adaptive like its generation
            return {
                "type": "enabled",
                "budget_tokens": int(self.settings["haiku_think_budget"]),
                "display": "summarized",
            }
        return {"type": "adaptive", "display": "summarized"}

    def _options(self, ctx: EpisodeContext) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            system_prompt=ctx.briefing,
            mcp_servers={
                "env": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(ctx.bridge_path)],
                    "env": ctx.bridge_env(),
                }
            },
            tools=[],  # no built-in tools: vanilla ReAct over the env only
            # No filesystem settings: without this the CLI walks up from cwd
            # and injects the repo CLAUDE.md into every session.
            setting_sources=[],
            # the CLI's own per-turn injections off (module docstring), the API
            # key blanked, the gateway redirect
            env=self._cli_env(),
            thinking=self._thinking_config(ctx),
            # the effort to send (base._effort): the seat word, else the harness
            # constant, else None = the CLI's own per-model default.
            effort=self._effort(ctx),
            betas=self._setting(ctx, "betas"),
            # ONLY our bridge — never the machine's global MCP config.
            strict_mcp_config=self.settings["strict_mcp_config"],
            # the tool names the session is opened with: the tool set's, in order.
            # Recorded, not a gate — under bypassPermissions registration is
            # the gate.
            allowed_tools=[f"mcp__env__{t}" for t in ctx.tool_set.allowed_tools],
            permission_mode=self.settings["permission_mode"],
            # look_around() returns four images in one MCP message; the default
            # 1 MiB stdout buffer truncates it and kills the session mid-parse
            max_buffer_size=int(self.settings["max_buffer_size"]),
            max_turns=ctx.max_turns,
            # Per-episode USD fuse. The CLI checks between
            # API calls and ends the session with subtype error_max_budget_usd
            # — whitelisted below as a scored truncation, like error_max_turns.
            max_budget_usd=ctx.max_budget_usd,
            model=ctx.model or None,
            cwd=str(ctx.workdir),
            # null = the SDK's bundled CLI (module docstring)
            cli_path=self.settings["cli_path"],
        )

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        result_msg: Any = None
        # Payload accounting — what the model ACTUALLY accumulated this
        # session, counted from the same stream record_raw sees. With
        # context_control=False the transcript only ever grows, so these are
        # totals, not a window.
        n_tool_results = 0
        n_images = 0
        n_state_blocks = 0
        init_tools: Any = None

        def record_raw(message: Any) -> None:
            sink.raw({"type": type(message).__name__, "msg": json_safe(message)})

        async with ClaudeSDKClient(options=self._options(ctx)) as client:
            # The CLI starts reasoning before MCP servers finish connecting;
            # gate the prompt on the bridge reporting 'connected'.
            bridge_status: str | None = None
            _deadline = float(self._setting(ctx, "bridge_connect_timeout_s"))
            _t0 = time.monotonic()
            while True:
                status = await client.get_mcp_status()
                entries = status.get("mcpServers", []) if isinstance(status, dict) else []
                bridge_status = next(
                    (e.get("status") for e in entries if e.get("name") == "env"), None
                )
                if (
                    bridge_status == "connected"
                    or bridge_status
                    in (
                        "failed",
                        "needs-auth",
                        "disabled",
                    )
                    or time.monotonic() - _t0 > _deadline
                ):
                    break
                await asyncio.sleep(0.5)
            sink.emit("bridge_status", {"status": bridge_status})
            if bridge_status != "connected":
                raise RuntimeError(f"env bridge not connected: {bridge_status}")

            await client.query(ctx.first_prompt)
            async for message in client.receive_response():
                record_raw(message)
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            sink.emit("assistant_text", {"text": block.text})
                        elif isinstance(block, ThinkingBlock):
                            sink.emit(
                                "thinking", {"chars": len(block.thinking), "text": block.thinking}
                            )
                        elif isinstance(block, ToolUseBlock):
                            sink.emit(
                                "tool_use",
                                {"id": block.id, "name": block.name, "input": block.input},
                            )
                elif isinstance(message, UserMessage):
                    content = message.content
                    blocks = content if isinstance(content, list) else []
                    for block in blocks:
                        if isinstance(block, ToolResultBlock):
                            texts = tool_result_texts(block.content)
                            n_tool_results += 1
                            n_images += sum(1 for t in texts if t == "<image elided>")
                            n_state_blocks += sum(1 for t in texts if "[STATE" in t)
                            sink.emit(
                                "tool_result", {"tool_use_id": block.tool_use_id, "texts": texts}
                            )
                elif isinstance(message, SystemMessage):
                    if getattr(message, "subtype", None) == "init":
                        data = getattr(message, "data", {}) or {}
                        init_tools = data.get("tools")
                        sink.emit(
                            "system_init", {"model": data.get("model"), "tools": data.get("tools")}
                        )
                elif isinstance(message, ResultMessage):
                    result_msg = message

        error = classify_outcome(result_msg)
        # roles/images/STATE copies the session accumulated and the tool list
        # the model was actually shown, under the full-history label.
        self._write_manifest(
            ctx,
            num_turns=getattr(result_msg, "num_turns", None),
            usage=json_safe(getattr(result_msg, "usage", None)),
            tool_results=n_tool_results,
            images=n_images,
            state_blocks=n_state_blocks,
            system_init_tools=init_tools,
        )
        return SessionOutcome(
            usage=json_safe(getattr(result_msg, "usage", None)),
            cost_usd=getattr(result_msg, "total_cost_usd", None),
            turns=getattr(result_msg, "num_turns", None),
            error=error,
            extra={"duration_ms": getattr(result_msg, "duration_ms", None)},
        )
