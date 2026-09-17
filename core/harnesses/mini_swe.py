"""mini-swe-agent adapter — the open ReAct loop.

Session block ported from the legacy driver (frozen in git history at
d10591e:beta-react-harness/run_episodes.py); the harness modules themselves
(model / env / nav_agent / bridge_toolset) are imported from
core/harnesses/mini/. The tool surface is the arm's bridge, as on the SDK and
codex seats — the harness carries no tools. litellm bills through the
provider API key.

Prompt delivery note: the shared driver hands us the RENDERED briefing;
mini's DefaultAgent runs its templates through jinja, so we wrap the text in
{% raw %} — rendering is then the identity function and the system prompt
stays byte-equal to the SDK cell (modulo jinja's stripped trailing newline,
the already-recorded difference).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from core.api import (
    ensure_serving,
    fake_gateway_for,
    provider_of,
    serving_audit,
    serving_log_offset,
)
from core.episode import EpisodeContext, EventSink, SessionOutcome

REPO_ROOT = Path(__file__).resolve().parents[2]  # the runner lives at the repo root
MINI_DIR = REPO_ROOT / "core" / "harnesses" / "mini"

# Serving for locally-hosted models lives in core/api (ollama.py, vllm.py),
# dispatched from the api entry — ensure_serving before episode 0,
# serving_audit in finalize, provider.default_api_base in _knobs.

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
if str(MINI_DIR) not in sys.path:
    sys.path.insert(0, str(MINI_DIR))


def _jinja_raw(text: str) -> str:
    return "{% raw %}" + text + "{% endraw %}"


class MiniSweAdapter:
    name = "mini-swe"

    def __init__(
        self,
        *,
        cost_limit: float = 5.0,
        image_window: int = 0,
        drop_params: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # the harness's own constants — configs/harness/mini.yaml beside its
        # _target_; the run cfg's extra overrides them by name.
        # ``model_kwargs``: litellm completion kwargs for every model on this
        # seat (temperature / max_tokens / top_p …); a model's own
        # ``params`` (models/models.yaml <name>: mini: {id, params}) lay on top.
        self.settings: dict[str, Any] = {
            "cost_limit": cost_limit,
            "image_window": image_window,
            "drop_params": drop_params,
            "model_kwargs": dict(model_kwargs or {}),
        }
        self.inherent: dict[str, Any] = {
            "auth": "provider API key (litellm billing)",
            "thinking": "not configured (plain completion)",
            "turn_cap": "hard (agent step_limit)",
            # mini's philosophy is "pack the full context every call"; the ONLY
            # bolt-on is image_window (model.py) = keep newest K frames, collapse
            # older to a text stub. K=0 (the current cell default) → full visual
            # history → linear request growth → the Anthropic cells can breach the
            # 32 MB request cap. This is a harness-inherent difference vs the SDK's
            # recent-window auto-eviction. See docs developer-guide/coding-agent/
            # harness-notes.
            "vision_context": "image_window=K newest frames (default 0 = full history, unbounded)",
        }
        self._log_offset = 0  # where this cell's stretch of the serve log starts
        self._model_id: str | None = None  # set in prepare(); keys the serving audit
        self._gateway = None  # the scripted fake endpoint (api=fake runs only)

    def prepare(self, spec) -> None:
        try:
            import minisweagent

            self.inherent["mini_version"] = getattr(minisweagent, "__version__", "?")
        except Exception:
            pass
        self._model_id = spec.model_id
        # api=fake: the scripted endpoint stands in for the vendor —
        # no key, no serving stack; litellm is pointed at it in _knobs. (The
        # litellm gateway of a cross-API seat means nothing here: mini speaks
        # litellm itself.)
        self._gateway = fake_gateway_for(spec, tag="mini")
        if self._gateway is not None:
            desc = self._gateway.describe()
            self.inherent["api_gateway"] = desc
            self.inherent["auth"] = desc["auth"]
            print(f"[mini] api gateway :{self._gateway.port} — {desc['backend']}")
            return
        self._check_auth(spec.model_id, spec.extra_dict)
        serving = ensure_serving(spec.model_id, spec.extra_dict)
        if serving:
            self.inherent["serving"] = serving
            self._log_offset = serving_log_offset(spec.model_id, spec.extra_dict)

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        """Serving-side audit for local cells (exact prompt-token counts for
        ollama, from its serve log) — delegated to the provider; {} for API
        cells. A missing audit reports itself rather than look like a clean one."""
        if self._gateway is not None:
            self._gateway.stop()
            self._gateway = None
        if "serving" not in self.inherent or self._model_id is None:
            return {}  # not a local cell — nothing to audit
        return serving_audit(self._model_id, self._log_offset)

    # ── auth ──

    @staticmethod
    def _is_local_model(model: str, extra: dict) -> bool:
        """Locally served model: an explicit api_base (ollama / vLLM / any
        OpenAI-compatible server) or a local litellm route prefix."""
        return bool(extra.get("api_base")) or any(
            model.startswith(p) for p in ("ollama", "hosted_vllm/", "openai/")
        )

    def _is_local(self, ctx: EpisodeContext) -> bool:
        return self._is_local_model(ctx.model, ctx.extra)

    def _check_auth(self, model_id: str, extra: dict) -> None:
        if self._is_local_model(model_id, extra):
            return  # local server — no provider key involved
        # provider registry (core/api) replaces the old name-substring
        # checks — same resolution order, same error shape, plus fallback
        # chains (a DashScope key riding OPENAI_API_KEY still resolves)
        provider = provider_of(model_id, extra)
        if provider is None or not provider.key_env:
            return
        if provider.resolve_key(extra)[0] is None:
            raise RuntimeError(
                f"none of {' / '.join(provider.key_chain(extra))} is set — "
                "litellm needs it (API billing)"
            )

    def _knobs(self, ctx: EpisodeContext) -> dict[str, Any]:
        set_cache_control = (
            "default_end"
            if any(
                s in (ctx.model or "").lower() for s in ["anthropic", "sonnet", "opus", "claude"]
            )
            else None
        )
        model_kwargs: dict[str, Any] = {
            "drop_params": self.settings["drop_params"],
            **self.settings["model_kwargs"],
        }
        if ctx.extra.get("api_base"):
            model_kwargs["api_base"] = ctx.extra["api_base"]
            provider = provider_of(ctx.model, ctx.extra)
            if provider is not None and provider.key_env:
                key, src = provider.resolve_key(ctx.extra)
                if key and src not in provider.key_fallbacks:
                    # the vendor's CANONICAL variable is set (DASHSCOPE_API_KEY
                    # et al) — pass the key explicitly so it stops riding
                    # OPENAI_API_KEY. Only-legacy-set shells keep the
                    # historical behavior byte-identically.
                    model_kwargs["api_key"] = key
        else:
            # local providers carry a default endpoint (ollama: OLLAMA_URL —
            # litellm must talk to the SAME server the adapter pinned; vllm:
            # VLLM_URL). Remote providers have none and this is a no-op.
            provider = provider_of(ctx.model, ctx.extra)
            if provider is not None and provider.default_api_base:
                model_kwargs["api_base"] = provider.default_api_base
        if ctx.effort != "default":
            # the run's effort (top-level `effort`) as litellm's generic
            # reasoning_effort — OpenAI takes it verbatim; for Anthropic
            # litellm maps it onto a thinking budget. "default" sends nothing.
            model_kwargs["reasoning_effort"] = ctx.effort
        # the model's own sampling params (models/models.yaml <name>: mini: {id, params})
        # — last, over the harness-wide model_kwargs
        model_kwargs.update(dict(ctx.extra.get("params") or {}))
        if self._gateway is not None:
            # the scripted endpoint: openai-chat on loopback whatever the model
            # id says (the provider is forced, not read off the name)
            model_kwargs.update(self._gateway.litellm_kwargs())
        return {
            "cost_limit": ctx.extra.get("cost_limit", self.settings["cost_limit"]),
            "image_window": ctx.extra.get("image_window", self.settings["image_window"]),
            "set_cache_control": set_cache_control,
            "model_kwargs": model_kwargs,
            # local servers have no litellm price entry — a hard cost lookup
            # would kill the run; tokens still land in the trajectory
            "cost_tracking": (
                "ignore_errors" if self._is_local(ctx) or self._gateway is not None else "default"
            ),
        }

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        knobs = self._knobs(ctx)
        return {
            "agent_config": {
                "step_limit": ctx.max_turns,
                "cost_limit": knobs["cost_limit"],
                "wall_time_limit_seconds": ctx.episode_timeout,
            },
            "effort": ctx.effort,
            "model_config": {
                "image_window": knobs["image_window"],
                "set_cache_control": knobs["set_cache_control"],
                "model_kwargs": knobs["model_kwargs"],
                "cost_tracking": knobs["cost_tracking"],
            },
        }

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        # mini's loop is synchronous; ride a thread so parallel workers overlap.
        return await asyncio.to_thread(self._run_sync, ctx, sink)

    def _run_sync(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from env import BridgeEnvironment
        from model import NavToolsModel
        from nav_agent import NavAgent

        knobs = self._knobs(ctx)  # auth + serving already settled in prepare()
        # the arm's own bridge, spawned exactly as the SDK / codex spawn it
        # (EH_EXECUTOR: who is driving, for the bridge's snapshot)
        env = BridgeEnvironment(
            bridge_path=str(ctx.bridge_path),
            env={**ctx.bridge_env(), "EH_EXECUTOR": "mini-swe"},
            server_url=ctx.server_url,
        )
        try:
            return self._drive(ctx, sink, knobs, env, NavToolsModel, NavAgent)
        finally:
            env.close()

    def _drive(
        self, ctx: EpisodeContext, sink: EventSink, knobs: dict[str, Any], env: Any, model_cls: Any, NavAgent: Any
    ) -> SessionOutcome:
        """One episode: the model over the env's toolset, mini's loop."""
        model = model_cls(
            model_name=ctx.model,
            tools=env.toolset.tool_schemas(),
            image_window=knobs["image_window"],
            set_cache_control=knobs["set_cache_control"],
            model_kwargs=knobs["model_kwargs"],
            cost_tracking=knobs["cost_tracking"],
        )
        sink_emit = sink.emit
        agent = NavAgent(
            model,
            env,
            system_template=_jinja_raw(ctx.briefing),
            instance_template=_jinja_raw(ctx.first_prompt),
            step_limit=ctx.max_turns,
            cost_limit=knobs["cost_limit"],
            wall_time_limit_seconds=ctx.episode_timeout,
            output_path=ctx.raw_dir / f"episode_{ctx.index}.traj.json",
            event_hook=sink_emit,
        )

        error: str | None = None
        exit_info: dict[str, Any] = {}
        try:
            exit_info = agent.run(task=ctx.instruction)
        except Exception as exc:
            error = repr(exc)
            sink.emit("driver_error", {"error": error})

        return SessionOutcome(
            usage={"billing": model.billing},  # per call, per model id (core.pricing); cost_usd = litellm's own meter
            cost_usd=round(agent.cost, 4),
            turns=agent.n_calls,
            error=error,
            extra={
                "exit_status": exit_info.get("exit_status"),
                "toolset_counts": dict(env.toolset.calls_by_tool),
                "toolset_env_steps": env.toolset.steps_taken,
                "toolset_end_reason": env.toolset.end_reason,
            },
        )
