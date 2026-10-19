"""mini-swe-agent adapter — the open ReAct loop, as a harness process.

mini runs as its own process per episode, like the SDK's Claude Code CLI and
codex: this adapter is a ``mip.harnesses.process.ProtocolAdapter`` that writes
the episode spec and starts ``mip/harnesses/mini/main.py``, which runs mini's loop
(model / env / nav_agent / bridge_toolset beside it) and speaks
``mip.harnesses.protocol``. The tool surface is the experiment's bridge, as on
the SDK and codex seats; the harness carries no tools. litellm bills through
the provider API key.

Prompt delivery: the runner hands over the RENDERED briefing; mini's
DefaultAgent runs its templates through jinja, so main.py wraps the text in
{% raw %}. Rendering is then the identity function and the system prompt is
byte-equal to the SDK seat's, except that jinja strips one trailing newline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mip.core.api import ModelRoute, ensure_serving, route_of, serving_audit, serving_log_offset
from mip.core.episode import EpisodeContext, ModelSpec
from mip.harnesses.base import Harness
from mip.harnesses.process import ProtocolAdapter
from mip.harnesses.protocol import EpisodeSpec

MINI_DIR = Path(__file__).resolve().parent / "mini"

# Serving for locally-hosted models lives in mip/core/api (ollama.py, vllm.py),
# dispatched from the api entry — ensure_serving before episode 0,
# serving_audit in finalize, the route's endpoint in _options.

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")


class MiniSweAdapter(ProtocolAdapter):
    name = "mini-swe"
    entry = MINI_DIR / "main.py"
    # sampling params exist only where the loop exposes them: mini's litellm kwargs
    accepts_params = True

    def __init__(
        self,
        *,
        effort: str | None = None,
        cost_limit: float = 5.0,
        image_window: int = 0,
        drop_params: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # the harness's own constants — mip/configs/harness/mini.yaml beside its
        # _target_; the run cfg's extra overrides them by name.
        # ``effort``: what the seat word `default` sends (null = nothing, the
        # provider's own default; a level = litellm's reasoning_effort constant).
        # ``model_kwargs``: litellm completion kwargs for every model on this
        # seat (temperature / max_tokens / top_p …); a model's own
        # ``params`` (mip/configs/models/models.yaml <name>: mini: {id, params}) lay on top.
        super().__init__(
            settings={
                "effort": effort,
                "cost_limit": cost_limit,
                "image_window": image_window,
                "drop_params": drop_params,
                "model_kwargs": dict(model_kwargs or {}),
            },
            harness=Harness(
                product="mini-swe-agent",
                # mini speaks litellm itself: only the scripted fake endpoint of an
                # api=fake run applies (the litellm gateway of a cross-API seat
                # means nothing here)
                wire="litellm",
                auth="provider API key (litellm billing)",
                thinking="not configured (plain completion)",
                turn_cap="hard (agent step_limit)",
                system_prompt="mini's system_template (the briefing, jinja raw-wrapped)",
                tool_surface="the tool set's bridge as litellm function tools (mini/env.py BridgeEnvironment)",
                # mini packs the full context on every call; the one addition is
                # image_window (model.py) = keep the newest K frames, collapse older
                # ones to a text stub. K=0 (the default) keeps the full visual
                # history, so requests grow linearly and an Anthropic request can
                # reach the 32 MB request cap.
                vision_context="image_window=K newest frames (default 0 = full history, unbounded)",
                context_control=False,
                context_label="mini-full-context (image_window=K newest frames)",
                effort_default="the provider's own default (litellm sends no reasoning_effort)",
                isolation=(
                    "a process per episode (mip/harnesses/mini/main.py); nothing on disk to keep out: "
                    "mini reads no user configuration; MSWEA_SILENT_STARTUP=1"
                ),
            ),
        )
        self._log_offset = 0  # where this cell's stretch of the serve log starts
        self._model_id: str | None = None  # set in _prepare(); keys the serving audit
        self._route: ModelRoute | None = None  # set in _prepare(): how the model is reached

    def _prepare(self, spec: ModelSpec) -> dict[str, Any]:
        self._model_id = spec.model_id
        self._route = route_of(spec.model_id, spec.extra_dict)
        if self._gateway is not None:
            # api=fake: the scripted endpoint stands in for the vendor — no
            # key, no serving stack; litellm is pointed at it in _options
            return {}
        self._route.require_key()
        serving = ensure_serving(spec.model_id, spec.extra_dict)
        if not serving:
            return {}
        self._log_offset = serving_log_offset(spec.model_id, spec.extra_dict)
        return {"serving": serving}

    def _version(self) -> str | None:
        try:
            import minisweagent
        except Exception:
            return None
        return getattr(minisweagent, "__version__", "?")

    def _finalize(self, run_dir: Path) -> dict[str, Any]:
        """Serving-side audit for local cells (exact prompt-token counts for
        ollama, from its serve log) — delegated to the provider; {} for API
        cells. A missing audit reports itself rather than look like a clean one."""
        if "serving" not in self.inherent or self._model_id is None:
            return {}  # not a local cell — nothing to audit
        return serving_audit(self._model_id, self._log_offset)

    def _options(self, ctx: EpisodeContext) -> dict[str, Any]:
        """What this episode hands mini: the agent's limits and the model's
        litellm kwargs (``_drive`` reads both)."""
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
        # the endpoint: the cell's api_base (with the vendor's canonical key beside it), or a
        # local vendor's default — litellm must talk to the SAME server the adapter pinned
        model_kwargs.update(self._route_of(ctx).litellm_kwargs())
        if (effort := self._effort(ctx)) is not None:
            # the effort to send (base._effort) as litellm's generic
            # reasoning_effort — OpenAI takes it verbatim; for Anthropic
            # litellm maps it onto a thinking budget. None sends nothing.
            model_kwargs["reasoning_effort"] = effort
        # the model's own sampling params (mip/configs/models/models.yaml <name>: mini: {id, params})
        # — last, over the harness-wide model_kwargs
        model_kwargs.update(dict(ctx.extra.get("params") or {}))
        if self._gateway is not None:
            # the scripted endpoint: openai-chat on loopback whatever the model
            # id says (the provider is forced, not read off the name)
            model_kwargs.update(self._gateway.litellm_kwargs())
        return {
            "agent": {
                "step_limit": ctx.max_turns,
                "cost_limit": self._setting(ctx, "cost_limit"),
                "wall_time_limit_seconds": ctx.episode_timeout,
            },
            "model": {
                "image_window": self._setting(ctx, "image_window"),
                "set_cache_control": set_cache_control,
                "model_kwargs": model_kwargs,
                # local servers have no litellm price entry — a hard cost lookup
                # would kill the run; tokens still land in the trajectory
                "cost_tracking": (
                    "ignore_errors"
                    if self._route_of(ctx).keyless or self._gateway is not None
                    else "default"
                ),
            },
        }

    def _route_of(self, ctx: EpisodeContext) -> ModelRoute:
        """The route resolved at prepare; resolved here when describe runs without one."""
        return self._route or route_of(ctx.model, ctx.extra)

    def _spec(self, ctx: EpisodeContext) -> EpisodeSpec:
        knobs = self._options(ctx)
        return {
            **self._base_spec(ctx),
            "model": {"id": ctx.model, "kwargs": dict(knobs["model"]["model_kwargs"])},
            "harness": {
                "cost_limit": knobs["agent"]["cost_limit"],
                "image_window": knobs["model"]["image_window"],
                "set_cache_control": knobs["model"]["set_cache_control"],
                "cost_tracking": knobs["model"]["cost_tracking"],
            },
        }
