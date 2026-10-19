"""mini-swe-agent as a harness process: ``python mip/harnesses/mini/main.py --episode <spec.json>``.

Started once per episode by ``mip/harnesses/mini_swe.py`` (``MiniSweAdapter``, a
``mip.harnesses.process.ProtocolAdapter``). It reads the episode spec, opens the experiment's
bridge (``env.BridgeEnvironment``), runs mini's loop (``nav_agent.NavAgent`` over
``model.NavToolsModel``) and speaks ``mip.harnesses.protocol`` on stdout: every message the
agent appends as an event line, the session's usage and counters as the outcome line, a session
that could not start as the fatal line.

mini's DefaultAgent runs its templates through jinja; the briefing arrives rendered, so it is
wrapped in ``{% raw %}`` and rendering is the identity function.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# mip.* from the repo root; this script's own folder (env, model, nav_agent) is already on
# sys.path as the script's directory
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")


def _jinja_raw(text: str) -> str:
    return "{% raw %}" + text + "{% endraw %}"


def _drive(spec: dict[str, Any], channel: Any, env: Any) -> None:
    from model import NavToolsModel
    from nav_agent import NavAgent

    knobs = spec["harness"]
    model = NavToolsModel(
        model_name=spec["model"]["id"],
        tools=env.toolset.tool_schemas(),
        image_window=knobs["image_window"],
        set_cache_control=knobs["set_cache_control"],
        model_kwargs=spec["model"]["kwargs"],
        cost_tracking=knobs["cost_tracking"],
    )
    agent = NavAgent(
        model,
        env,
        system_template=_jinja_raw(spec["briefing"]),
        instance_template=_jinja_raw(spec["first_prompt"]),
        step_limit=spec["max_turns"],
        cost_limit=knobs["cost_limit"],
        wall_time_limit_seconds=spec["timeout_s"],
        output_path=Path(spec["raw_dir"]) / f"episode_{spec['index']}.traj.json",
        event_hook=channel.emit,
    )
    error: str | None = None
    exit_info: dict[str, Any] = {}
    try:
        exit_info = agent.run(task=spec["instruction"])
    except Exception as exc:
        error = repr(exc)
        channel.emit("driver_error", {"error": error})
    channel.outcome(
        usage={"billing": model.billing},  # per call, per model id (mip.core.pricing)
        cost_usd=round(agent.cost, 4),  # litellm's own meter
        turns=agent.n_calls,
        error=error,
        extra={
            "exit_status": exit_info.get("exit_status"),
            "toolset_counts": dict(env.toolset.calls_by_tool),
            "toolset_env_steps": env.toolset.steps_taken,
            "toolset_end_reason": env.toolset.end_reason,
        },
    )


def main() -> int:
    from mip.harnesses.protocol import Channel, read_spec

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--episode", required=True, help="the episode spec (JSON)")
    args = parser.parse_args()
    channel = Channel()
    spec = read_spec(args.episode)
    try:
        from env import BridgeEnvironment

        tools = spec["tools"]
        env = BridgeEnvironment(
            bridge_path=tools["args"][0], env=tools["env"], server_url=spec["server_url"]
        )
    except Exception as exc:
        channel.fatal(repr(exc))
        return 1
    try:
        _drive(spec, channel, env)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
