"""goto(waypoint) — walk to one candidate of the LATEST look via the env's step_hightolow
(heading write + FORWARD primitives, charged to the step budget). The move invalidates the
numbering: the next goto needs a new look — unless ``auto_observe`` is on, when the result
itself carries the new panorama and numbering."""

from __future__ import annotations

import json
from typing import Any

from mip.core.toolset import Session
from mip.tools.waypoint.panorama import (
    ENV_TIMEOUT_S,
    NEEDS_WP_SERVER,
    budget_fields,
    candidates,
    direction_of,
    max_moves,
    move_fields,
    render_and_predict,
)

__all__ = ["DESCRIPTION", "NAME", "NEEDS_WP_SERVER", "describe", "register"]
NAME = "goto"
DESCRIPTION = (
    "Move to one numbered waypoint from the LATEST {look}() result: the robot turns "
    "toward it and walks there. Moving invalidates the old numbers — call {look}() again "
    "afterwards to see the new surroundings and waypoints. Returns steps consumed, "
    "remaining budget, and whether the episode is over."
)


def describe(slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    """``{look}``: the set's panorama tool — observe_waypoints beside the primitives, else observe."""
    return DESCRIPTION.format(
        look="observe_waypoints" if "observe_waypoints" in names else "observe"
    )


def register(mcp: Any, s: Session, description: str) -> None:
    from mcp.server.fastmcp import Image

    # the look that numbers the candidates: the Hybrid set's observe_waypoints, else the Waypoint set's observe
    look = "observe_waypoints" if "waypoint.observe_waypoints" in s.tools else "observe"
    auto_observe = bool(s.options.get("auto_observe"))
    # ``legacy_dashes`` (the SLAM sets): this message is worded with "--"
    dash = "--" if s.options.get("legacy_dashes") else "—"

    def walk(waypoint: Any) -> Any:
        s.tool_calls += 1
        if s.episode_over:
            return {"error": f"episode already over ({s.end_reason}); no more moves possible"}
        cands = candidates(s)
        if cands is None:
            return {"error": f"no current waypoint set {dash} call {look}() first"}
        try:
            waypoint = int(waypoint)
        except (TypeError, ValueError):
            return {"error": f"invalid waypoint {waypoint!r}; expected an integer"}
        if not 1 <= waypoint <= len(cands):
            return {
                "error": (
                    f"invalid waypoint {waypoint}; valid choices are 1-{len(cands)} from the LATEST {look}()"
                )
            }

        cand = cands[waypoint - 1]
        outputs = s.call(
            "step_hightolow", timeout=ENV_TIMEOUT_S, angle=cand["angle"], distance=cand["distance"]
        )
        if "error" in outputs and "info" not in outputs:
            return {"error": f"step_hightolow: {outputs['error']}"}
        info = outputs.get("info") or {}
        if isinstance(info.get("step_count"), (int, float)):
            s.steps_taken = int(info["step_count"])
        for listener in s.step_listeners:
            listener.begin()
            listener.observe(info)
        s.notes["wp_moves"] += 1
        s.moved()
        terminated = bool(outputs.get("terminated"))
        truncated = bool(outputs.get("truncated"))
        if terminated or truncated:
            s.episode_over = True
            s.end_reason = "step_budget_exhausted" if truncated else "terminated"
        elif max_moves(s) > 0 and s.notes["wp_moves"] >= max_moves(s):
            # decision-step budget spent — truncate like habitat's step cap
            s.episode_over = True
            s.end_reason = "wp_move_budget_exhausted"

        result: dict[str, Any] = {
            "moved_to": waypoint,
            "direction": direction_of(cand["angle"]),
            "distance_m": round(cand["distance"], 2),
            "steps_taken_total": s.steps_taken,
            "episode_over": s.episode_over,
            "end_reason": s.end_reason,
            **move_fields(s),
            **budget_fields(s),
        }
        log: dict[str, Any] = {}
        for listener in s.step_listeners:
            result.update(listener.result_fields())
            log.update(listener.log_fields("goto"))
        s.live_log(
            {
                "goto": waypoint,
                "distance_m": result["distance_m"],
                "steps_taken_total": s.steps_taken,
                **log,
                "episode_over": s.episode_over,
            }
        )
        if s.episode_over or not auto_observe:
            # the position changed, so the numbering is stale until the next look
            if not s.episode_over:
                result["note"] = f"call {look}() to see the new surroundings and waypoints"
            return result

        # auto-observe: re-render and re-predict for the new position, so the next
        # goto() can pick a number straight from THIS result
        png, obs_status = render_and_predict(
            s,
            remaining=False,
            empty_note=(
                "no reachable waypoints predicted here; if you believe you are at "
                "the goal call stop(), otherwise goto() a prior waypoint or make "
                "another move"
            ),
            log_key="observe_wp",
        )
        result["waypoints"] = obs_status["waypoints"]
        result["action_options"] = obs_status["action_options"]
        if "note" in obs_status:
            result["note"] = obs_status["note"]
        result["new_view"] = "attached panorama + waypoints are for your NEW position"
        return [Image(data=png, format="png"), json.dumps(result)]

    # The return annotation is part of the wire: a dict annotation makes FastMCP send the
    # result as structuredContent too. The Hybrid sets register goto typed (typed_results), the
    # Waypoint set untyped (its auto-observe result is a list).
    if s.options.get("typed_results"):

        def goto(waypoint: int) -> dict[str, Any]:
            return walk(waypoint)

    else:

        def goto(waypoint: int) -> list:  # a bare `list` return type: FastMCP sends unstructured content only
            return walk(waypoint)

    mcp.tool(description=description)(goto)
