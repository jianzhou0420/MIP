"""MCP bridge — the bare two-tool surface over EmbodiedScore-envs, every line.

bareES (2026-09-08): bare's bridge (``exp_workspace/bare/bridge.py``) re-pointed
at the ``env_bare_es`` nodeset — the frozen copy of ``env_embodiedscore`` —
so ONE bridge serves all thirteen package lines. The agent sees the bare
surface, shaped by the line's task (``BAREES_TASK``, set by the driver from
what reset() returns):

- ``observe()``      -> egocentric RGB (base64 PNG passthrough as MCP image)
- ``step(actions)``  -> discrete actions on the STANDARD table: 0 STOP ·
                        1 FORWARD 0.25 m · 2 LEFT 15° · 3 RIGHT 15° ·
                        4 LOOK_UP 30° · 5 LOOK_DOWN 30° (pitch clamped ±60°);
                        GOAT adds 6 SUBTASK_STOP (close the current sub-goal);
                        the EQA lines have no STOP (answer() ends the episode)
- ``answer(text)``   -> eqa / express only: record the answer and END the
                        episode (a letter on the multiple-choice lines, free
                        text on EXPRESS); rides ``env_bare_es__submit_answer``

Task shapes (BAREES_TASK):
    nav      VLN-CE R2R / RxR, IVLN-CE, VLNverse — instruction in the briefing, STOP terminal
    objnav   ObjectNav HM3D/MP3D, OVON — category in the briefing, STOP terminal
    goat     GOAT-Bench — 5-10 ordered sub-goals; step([6]) closes the current one
             and the step result announces the next (text, or the target's
             photo as an attached image — goat-bench renders it from the stored
             camera pose, the dataset ships no JPEG). Only the CURRENT sub-goal
             is ever shown, as upstream's current-subtask sensor does. The
             first sub-goal's photo rides the first observe() of the episode.
    eqa      HM-EQA / MT-HM3D — question + choices in the briefing, answer(letter)
    express  EXPRESS-Bench — free-form question, answer(text); judged driver-side

Deviations from bare's bridge, all surfaced to the model: the STANDARD
action table (6 actions, tilt included) on every line instead of bare's
0-3; the Isaac lines (VLNverse) refuse LOOK actions (the worker renders
yaw-only poses) — their table is 0-3 and the descriptions say so.

Episode selection and metric collection stay driver-side; the agent never
sees SR/SPL, reward, pose, depth, or panoramas. One bridge process serves
one agent session = one episode, so per-episode step accounting lives in
module globals.

Env vars (driver-rendered from arm.yaml "surface"):
    BAREES_SERVER_URL    the env server (core.envserver) base URL (default http://127.0.0.1:9200)
    BAREES_TASK          nav | objnav | goat | eqa | express (default nav)
    BAREES_ACTIONS       the line's action count (4 = Isaac lines, 6 = STANDARD,
                         7 = GOAT); default 6
    BAREES_STEP_BUDGET   advisory low-level step budget echoed to the agent
                         (the env truncates authoritatively regardless)
    BAREES_TURN_BUDGET   the driver's max_turns; >0 turns on the per-step
                         tool-call broadcast (off in the bare condition)
    BAREES_BARE          "1" = bare toolface (no clearance, no look_around) — always 1 on this arm
    BAREES_AUTO_OBSERVE  "1" = step() attaches the post-move view (off: classic alternation)
    BAREES_LIVE_DIR      optional dir for live spectating (obs_NNNN.png, actions.log)
"""

from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

SERVER_URL = os.environ.get("BAREES_SERVER_URL", "http://127.0.0.1:9200")
TASK = os.environ.get("BAREES_TASK", "nav")
N_ACTIONS = int(os.environ.get("BAREES_ACTIONS", "6"))
STEP_BUDGET = int(os.environ.get("BAREES_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("BAREES_TURN_BUDGET", "0"))
PANO_VIEW_PX = int(os.environ.get("BAREES_PANO_VIEW_PX", "0"))
LIVE_DIR = Path(os.environ["BAREES_LIVE_DIR"]) if os.environ.get("BAREES_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50
BARE = os.environ.get("BAREES_BARE", "1") == "1"
AUTO_OBSERVE = os.environ.get("BAREES_AUTO_OBSERVE") == "1"

if TASK not in ("nav", "objnav", "goat", "eqa", "express"):
    raise SystemExit(f"BAREES_TASK must be nav|objnav|goat|eqa|express, got {TASK!r}")
EQA = TASK in ("eqa", "express")
GOAT = TASK == "goat"

# The line's action table: a prefix of 0 STOP · 1 FWD · 2 LEFT · 3 RIGHT ·
# 4 LOOK_UP · 5 LOOK_DOWN · 6 SUBTASK_STOP. The EQA lines drop STOP (answer()
# is the terminal); GOAT keeps 6.
_TABLE = list(range(N_ACTIONS))
VALID_ACTIONS = [a for a in _TABLE if not (EQA and a == 0)]
_NAMES = {
    0: "STOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "LOOK_UP",
    5: "LOOK_DOWN",
    6: "SUBTASK_STOP",
}

_MOVES = "1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees"
_TILT = (
    ", 4 = tilt the camera up 30 degrees, 5 = tilt the camera down 30 degrees "
    "(tilt changes the camera pitch only, not your position or heading; pitch is clamped to ±60°)"
    if 5 in VALID_ACTIONS
    else ""
)
_STOP_CLAUSE = {
    "nav": "0 = STOP (permanently ENDS the episode — issue it only when you believe the robot is within 3 meters of the goal), ",
    "objnav": "0 = STOP (permanently ENDS the episode — issue it only when you can see the target and are standing right next to it), ",
    "goat": "0 = STOP (permanently ENDS the whole episode — normally you never need it: closing the last sub-goal ends the episode), ",
    "eqa": "",
    "express": "",
}[TASK]
_GOAT_CLAUSE = (
    ", 6 = SUBTASK_STOP (declare the CURRENT sub-goal reached — issue it only when you are standing "
    "right next to it, within 0.25 m; the result then announces the next sub-goal, with its photo "
    "attached when the sub-goal is given as an image)"
    if GOAT
    else ""
)
_NO_STOP = " There is NO stop action — the episode ends when you call answer()." if EQA else ""

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view. Pure read — does not advance the simulator or "
    "consume step budget."
    + (
        " On this benchmark the very first observe() of the episode also attaches the current "
        "sub-goal's target photo when that sub-goal is an image."
        if GOAT
        else ""
    )
)
_STEP_DESC = (
    "Execute a sequence of movement actions, in order.\n\n"
    f"Actions: {_STOP_CLAUSE}{_MOVES}{_TILT}{_GOAT_CLAUSE}.{_NO_STOP}\n\n"
    "Executes sequentially and halts early if the episode ends. Returns how "
    "many actions ran, total steps taken, remaining budget, and whether the "
    "episode is over."
    + (
        " The resulting camera view is attached as an image (auto-observe), so "
        "you do NOT need to call observe() after moving — only observe() for "
        "your very first look. A finished episode carries no image."
        if AUTO_OBSERVE
        else " The camera view changes after stepping — call observe() to see the result."
    )
)
_ANSWER_DESC = (
    "Permanently END the episode by answering the multiple-choice question. "
    'Pass a single letter: "A", "B", "C" or "D". This is your one '
    "and only answer — call it once you have seen enough of the building to "
    "be confident. An episode that ends without answer() scores zero."
    if TASK == "eqa"
    else "Permanently END the episode by answering the question in free-form "
    "natural language (one or two sentences). Your answer is judged on its "
    "correctness AND on whether your final camera view supports it, so "
    "answer while the relevant object or place is in view. This is your one "
    "and only answer; an episode that ends without answer() scores zero."
)

mcp = FastMCP("bare-es-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_episode_over = False
_end_reason: str | None = None
_answer: str | None = None
_goal_shown = False  # GOAT: whether the current sub-goal's photo has been delivered
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn (off when
    TURN_BUDGET is 0, i.e. in the bare condition)."""
    if TURN_BUDGET <= 0:
        return {}
    remaining = max(0, TURN_BUDGET - _tool_calls)
    fields: dict[str, Any] = {"tool_calls_used": _tool_calls, "tool_calls_remaining": remaining}
    terminal = "call answer()" if EQA else "call step([0])"
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


def _clearance_m(depth_field: dict[str, Any] | None) -> dict[str, float] | None:
    """Metric free-space readout from the raw depth frame (non-bare only)."""
    if not isinstance(depth_field, dict) or "__ndarray__" not in depth_field:
        return None
    try:
        arr = np.frombuffer(
            base64.b64decode(depth_field["__ndarray__"]), dtype=depth_field.get("dtype", "float32")
        ).reshape(depth_field["shape"])
    except Exception:
        return None
    h, w = arr.shape[:2]
    band = arr[int(h * 0.40) : int(h * 0.65), :]
    sectors = {
        "left": band[:, : w // 3],
        "center": band[:, w // 3 : 2 * w // 3],
        "right": band[:, 2 * w // 3 :],
    }
    return {
        name: round(float(np.percentile(sector, 10)) * 10.0, 1) for name, sector in sectors.items()
    }


def _downscale(png: bytes, side: int) -> bytes:
    img = PILImage.open(BytesIO(png))
    if not side or max(img.size) <= side:
        return png
    img = img.resize((side, side), PILImage.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _live_frame(png: bytes) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / f"obs_{_obs_count:04d}_step{_steps_taken:03d}.png").write_bytes(png)
    (LIVE_DIR / "latest.png").write_bytes(png)


def _live_log(entry: dict[str, Any]) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with (LIVE_DIR / "actions.log").open("a") as fh:
        fh.write(json.dumps({"t": round(time.time() - _t0, 1), **entry}) + "\n")


def _call(verb: str, **kwargs: Any) -> dict[str, Any]:
    """One env verb (core.envserver: POST /<verb>, kwargs as JSON). Kept
    inline — the bridge is frozen arm code and imports nothing of the repo."""
    resp = requests.post(f"{SERVER_URL}/{verb}", json=kwargs, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _capture_view() -> tuple[bytes, dict[str, float] | None]:
    """Render the current egocentric RGB (+ clearance when not bare). Pure
    read — advances the live-frame counter, never the simulator."""
    global _obs_count
    outputs = _call("observe")
    png = base64.b64decode(outputs["rgb"])
    _obs_count += 1
    _live_frame(png)
    clearance = None if BARE else _clearance_m(outputs.get("depth"))
    return png, clearance


# ── GOAT: the current sub-goal, announced through the tool-result channel ──


def _goal_content(prefix: str) -> list:
    """Text (+ photo) describing the sub-goal in force, from the nodeset's
    observe_goal verb. Only the current one is ever exposed."""
    global _goal_shown
    g = _call("goal_spec")
    goal = g.get("goal") or {}
    kind = goal.get("kind", "none")
    k, n = int(g.get("goal_index", 0)) + 1, int(g.get("n_goals", 1))
    if g.get("episode_over") or kind == "none":
        return [f"{prefix} all {n} sub-goals are closed — the episode is over."]
    if kind == "object":
        text = (
            f'{prefix} sub-goal {k} of {n}: find a "{goal.get("category")}" (any instance counts).'
        )
        return [text]
    if kind == "description":
        text = f'{prefix} sub-goal {k} of {n}: find the object this describes — "{goal.get("description")}".'
        return [text]
    if kind == "image":
        text = (
            f"{prefix} sub-goal {k} of {n}: find the specific object shown in the attached photo "
            "(the photo was taken inside this building; find that very instance)."
        )
        img = g.get("goal_image")
        _goal_shown = True
        if img:
            return [text, Image(data=base64.b64decode(img), format="png")]
        return [text + " (photo unavailable)"]
    return [f"{prefix} sub-goal {k} of {n}: {g.get('task', '')}"]


@mcp.tool(description=_OBSERVE_DESC)
def observe() -> list:
    global _tool_calls
    _tool_calls += 1
    png, clearance = _capture_view()
    content: list[Any] = [Image(data=png, format="png")]
    if GOAT and not _goal_shown:
        content.extend(_goal_content("Current"))
    if not BARE:
        content.append(json.dumps({"clearance_m": clearance, **_budget_fields()}))
    return content


def look_around() -> list:
    """Four labeled views (ahead/right/behind/left) in one call; rotates 360°
    and restores the heading. Not registered on the bare arm."""
    global _obs_count, _steps_taken, _episode_over, _end_reason, _tool_calls
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more steps possible"]
    content: list[Any] = []
    for label in ("ahead (0°)", "right (+90°)", "behind (+180°)", "left (+270°)"):
        outputs = _call("observe")
        png = base64.b64decode(outputs["rgb"])
        _obs_count += 1
        _live_frame(png)
        content.extend([label, Image(data=_downscale(png, PANO_VIEW_PX), format="png")])
        for _ in range(6):
            outputs = _call("step", action=3)
            _steps_taken += 1
            if outputs.get("terminated") or outputs.get("truncated"):
                _episode_over, _end_reason = True, "step_budget_exhausted"
                break
        if _episode_over:
            break
    status = {
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "heading_restored": not _episode_over,
        **_budget_fields(),
    }
    content.append(json.dumps(status))
    _live_log({"look_around": True, **status})
    return content


if not BARE:
    mcp.tool()(look_around)


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> list:  # bare `list` => FastMCP unstructured path
    global _tool_calls
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more steps possible"}
    if not actions:
        return {"error": "empty action list"}
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    bad = [a for a in actions if a not in VALID_ACTIONS]
    if bad:
        valid = " ".join(f"{a}={_NAMES[a]}" for a in VALID_ACTIONS)
        return {"error": f"invalid actions {bad}; valid: {valid}"}
    return _with_view(_execute_actions(actions))


def _execute_actions(actions: list[int]) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason, _goal_shown
    executed = 0
    subtask_closed = False
    for action in actions:
        outputs = _call("step", action=action)
        executed += 1
        _steps_taken += 1
        info = outputs.get("info") or {}
        if "error" in info:
            _live_log({"actions": actions, "error": info["error"]})
            return {
                "error": info["error"],
                "executed": executed - 1,
                "requested": len(actions),
                "steps_taken_total": _steps_taken,
            }
        if info.get("subtask_closed"):
            subtask_closed = True
            _goal_shown = False
        terminated = bool(outputs.get("terminated"))
        truncated = bool(outputs.get("truncated"))
        if terminated or truncated:
            _episode_over = True
            if action == 0:
                _end_reason = "stop_called"
            elif truncated:
                _end_reason = "step_budget_exhausted"
            elif GOAT and action == 6:
                _end_reason = "subtasks_closed"
            else:
                _end_reason = "terminated"
            break
        if subtask_closed:
            break  # announce the new sub-goal before running any further actions
    result = {
        "executed": executed,
        "requested": len(actions),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_budget_fields(),
    }
    if subtask_closed:
        result["subtask_closed"] = True
        if executed < len(actions):
            result["note"] = (
                f"stopped after action {executed} of {len(actions)}: a sub-goal was closed, "
                "read the next sub-goal below before continuing"
            )
    _live_log({"actions": actions, **result})
    return result


def _with_view(result: dict[str, Any]) -> Any:
    """Attach the post-move view (auto-observe) and, on GOAT, the next
    sub-goal after a SUBTASK_STOP. A finished episode carries no view."""
    if "error" in result:
        return result
    content: list[Any] = []
    if GOAT and result.get("subtask_closed"):
        content.extend(_goal_content("Next") if not _episode_over else _goal_content("Done:"))
    if _episode_over:
        return [json.dumps(result), *content] if content else result
    if not AUTO_OBSERVE:
        if not BARE:
            _, clearance = _capture_view()
            if clearance:
                result["clearance_m"] = clearance
        return [json.dumps(result), *content] if content else result
    png, clearance = _capture_view()
    if clearance:
        result["clearance_m"] = clearance
    result["new_view"] = "the attached image is the camera view AFTER these actions"
    return [Image(data=png, format="png"), json.dumps(result), *content]


def answer(text: str) -> dict[str, Any]:
    """Permanently END the episode by answering the question (a letter on
    the multiple-choice lines, free text on EXPRESS)."""
    global _tool_calls, _episode_over, _end_reason, _answer, _steps_taken
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); your answer was {_answer!r}"}
    clean = str(text or "").strip()
    if TASK == "eqa":
        clean = clean.strip("\"'.()").upper()
        if clean not in ("A", "B", "C", "D"):
            return {"error": f'invalid answer {text!r}; pass a single letter: "A", "B", "C" or "D"'}
    elif not clean:
        return {"error": "empty answer; write one or two sentences"}
    outputs = _call("submit_answer", answer=clean)
    info = outputs.get("info") or {}
    if "error" in info:
        return {"error": info["error"]}
    _steps_taken += 1  # submit_answer is the STOP step on the discrete lines
    _episode_over = True
    _end_reason = "answer_submitted"
    _answer = clean
    # steps_taken_total makes this the EventSink's last parsed step result,
    # which is how the driver reads the answer back for evaluate().
    result = {
        "answer": clean,
        "answer_submitted": True,
        "episode_over": True,
        "end_reason": _end_reason,
        "steps_taken_total": _steps_taken,
        "message": f"Final answer {clean!r} recorded — the episode is over.",
    }
    _live_log(result)
    return result


if EQA:
    mcp.tool(description=_ANSWER_DESC)(answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
