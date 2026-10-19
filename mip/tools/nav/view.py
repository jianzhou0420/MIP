"""What the navigation tools share: the action table, the camera view, the step loop and the
turn-budget broadcast — all over one Session."""

from __future__ import annotations

import base64
import json
from typing import Any

from mip.core.toolset import Session

TASKS = ("nav", "objnav", "eqa")
MAX_ACTIONS_PER_CALL = 50
NAMES = {
    0: "STOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "LOOK_UP",
    5: "LOOK_DOWN",
}


def check_task(s: Session) -> None:
    if s.task not in TASKS:
        raise SystemExit(f"the nav tools serve {'|'.join(TASKS)}, got task {s.task!r}")


def is_eqa(s: Session) -> bool:
    return s.task == "eqa"


def valid_actions(s: Session) -> list[int]:
    """The line's action table, minus STOP on the question lines."""
    n = int(s.n_actions) if s.n_actions is not None else 6
    return [a for a in range(n) if not (is_eqa(s) and a == 0)]


def budget_fields(s: Session) -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn (off under the ``bare`` switch)."""
    remaining = s.calls_remaining()
    if remaining is None:
        return {}
    fields: dict[str, Any] = {"tool_calls_used": s.tool_calls, "tool_calls_remaining": remaining}
    terminal = "call answer()" if is_eqa(s) else "call step([0])"
    if remaining <= 10:
        fields["BUDGET_WARNING"] = (
            f"CRITICAL — only {remaining} tool calls left before this session is "
            f"killed. Execute your terminal protocol NOW: {terminal}. Ending without it scores ZERO."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is killed. "
            "Stop exploring new areas; commit to your best candidate and finish before the budget runs out."
        )
    return fields


def capture_view(s: Session) -> bytes:
    """Render the current egocentric RGB. Pure read — advances the live-frame
    counter, never the simulator."""
    outputs = s.call("observe")
    png = base64.b64decode(outputs["rgb"])
    s.obs_count += 1
    s.live_write(f"obs_{s.obs_count:04d}_step{s.steps_taken:03d}.png", png)
    return png


def over_error(s: Session) -> dict[str, Any] | None:
    if s.episode_over:
        return {"error": f"episode already over ({s.end_reason}); no more steps possible"}
    return None


def action_error(s: Session, actions: list[int]) -> dict[str, Any] | None:
    """The error a non-empty step() call gets before anything runs, or None."""
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    valid = valid_actions(s)
    bad = [a for a in actions if a not in valid]
    if bad:
        table = " ".join(f"{a}={NAMES[a]}" for a in valid)
        return {"error": f"invalid actions {bad}; valid: {table}"}
    return None


def execute_actions(s: Session, actions: list[int]) -> dict[str, Any]:
    executed = 0
    for listener in s.step_listeners:
        listener.begin()
    for action in actions:
        outputs = s.call("step", action=action)
        executed += 1
        s.steps_taken += 1
        info = outputs.get("info") or {}
        if "error" in info:
            s.live_log({"actions": actions, "error": info["error"]})
            return {
                "error": info["error"],
                "executed": executed - 1,
                "requested": len(actions),
                "steps_taken_total": s.steps_taken,
            }
        for listener in s.step_listeners:
            listener.observe(info)
        terminated = bool(outputs.get("terminated"))
        truncated = bool(outputs.get("truncated"))
        if terminated or truncated:
            s.episode_over = True
            if action == 0:
                s.end_reason = "stop_called"
            elif truncated:
                s.end_reason = "step_budget_exhausted"
            else:
                s.end_reason = "terminated"
            break
    s.moved()
    result = {
        "executed": executed,
        "requested": len(actions),
        "steps_taken_total": s.steps_taken,
        "steps_remaining_approx": max(0, s.step_budget - s.steps_taken),
        "episode_over": s.episode_over,
        "end_reason": s.end_reason,
        **budget_fields(s),
    }
    log: dict[str, Any] = {"actions": actions}
    for listener in s.step_listeners:
        result.update(listener.result_fields())
        log.update(listener.log_fields("step"))
    s.live_log({**log, **result})
    return result


def with_view(s: Session, result: dict[str, Any], *, auto_observe: bool) -> Any:
    """Attach the post-move view (auto-observe). A finished episode carries no view."""
    from mcp.server.fastmcp import Image

    if "error" in result:
        return result
    if s.episode_over or not auto_observe:
        return result
    png = capture_view(s)
    result["new_view"] = "the attached image is the camera view AFTER these actions"
    return [Image(data=png, format="png"), json.dumps(result)]
