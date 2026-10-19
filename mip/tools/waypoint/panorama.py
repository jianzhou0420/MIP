"""What the waypoint tools share: the predictor call, the candidate list (set by a look,
cleared by any move), the annotated [Left|Front|Right|Back] strip, and the move and turn
budget broadcasts."""

from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Any

from mip.core.toolset import Session

NEEDS_WP_SERVER = True
N_PANO_VIEWS = 12
# dir_id -> strip slot: the env's panorama dir i faces i*30 deg counter-
# clockwise, so 3 = Left, 0 = Front, 9 = Right, 6 = Back — the
# [Left|Front|Right|Back] display order at zero extra render cost.
STRIP_DIRS = ((3, "Left"), (0, "Front"), (9, "Right"), (6, "Back"))
ENV_TIMEOUT_S = 600  # a 12-view panorama renders slowly
#: the look's description — the Waypoint set's observe and the Hybrid set's observe_waypoints alike
DESCRIPTION = (
    "Look around from where you stand. Returns a panoramic image of four views labeled "
    "Left / Front / Right / Back with numbered green circles marking the waypoints you "
    "can move to, plus a JSON listing each waypoint's direction, angle (degrees left of "
    "your heading; negative = right) and distance in meters. Pure read — does not "
    "advance the simulator or consume step budget."
)


def preflight(*, wp_server: str | None, **_facts: Any) -> None:
    """Before episode 0 (mip.core.toolset.ToolSet.preflight): the waypoint tools refuse a run
    whose predictor is missing — the tools would degrade SILENTLY."""
    if not wp_server:
        raise RuntimeError("the waypoint tools need run.wp_server (waypoint-predictor auto_host)")
    print(f"[std] {wp_server} healthy: {predictor_health(wp_server)['name']} (waypoint predictor)")


def predictor_health(wp_server: str) -> dict[str, Any]:
    """The predictor's /health; raises when it does not answer."""
    import requests

    health = requests.get(f"{wp_server}/health", timeout=10)
    health.raise_for_status()
    return health.json()


def candidates(s: Session) -> list[dict[str, float]] | None:
    """The numbered candidates of the latest look; None = a look is required."""
    return s.notes.get("wp_candidates")


def arm_on_moves(s: Session) -> None:
    """Every move (a goto, a primitive step) makes the numbering stale."""
    s.notes.setdefault("wp_candidates", None)
    s.notes.setdefault("wp_moves", 0)
    s.on_move(lambda: s.notes.__setitem__("wp_candidates", None))


def max_moves(s: Session) -> int:
    return int(s.options.get("wp_max_moves") or 0)


def move_fields(s: Session) -> dict[str, Any]:
    """Waypoint-move budget broadcast (off when the cap is 0: nothing about moves reaches the model)."""
    cap = max_moves(s)
    if cap <= 0:
        return {}
    moves = s.notes["wp_moves"]
    remaining = max(0, cap - moves)
    fields: dict[str, Any] = {"moves_used": moves, "moves_remaining": remaining}
    if 0 < remaining <= 3:
        fields["MOVE_WARNING"] = (
            f"Only {remaining} waypoint move(s) left before the episode ends. "
            "If you are at the goal, call stop() now; otherwise make them count."
        )
    return fields


def budget_fields(s: Session) -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn; empty when the broadcast is off."""
    remaining = s.calls_remaining()
    if remaining is None:
        return {}
    fields: dict[str, Any] = {
        "tool_calls_used": s.tool_calls,
        "tool_calls_remaining": remaining,
    }
    if remaining <= 10:
        fields["BUDGET_WARNING"] = (
            f"CRITICAL — only {remaining} tool calls left before this session is "
            "killed. Move to your best goal candidate (or stay right here if it "
            "scores best) and call stop() NOW. Ending without stop scores ZERO."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is killed. "
            "Stop exploring new areas; commit to your best goal candidate, "
            "approach it, and stop() before the budget runs out."
        )
    return fields


def call_wp(s: Session, function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """One predictor call (POST /call/{fn} with the ``inputs`` envelope, ``outputs`` back)."""
    import requests

    url = s.wp_server_url or "http://127.0.0.1:9210"
    resp = requests.post(f"{url}/call/{function_name}", json={"inputs": inputs}, timeout=600)
    resp.raise_for_status()
    return resp.json()["outputs"]


def normalize_candidates(raw: Any) -> list[dict[str, float]]:
    """Predictor output -> ordered [{angle, distance}]; marker text is i+1.

    smartway_waypoint__predict: {"0": {angle, distance, rgb_base64}, ...}
    opennav_waypoint__predict:  {"slot": [angle_rad, distance_m], ...}
    Both emit dict entries in ascending angle-index order (the waypoint
    numbering); JSON object order survives the HTTP round trip, so response
    order is kept as-is.
    """
    out: list[dict[str, float]] = []
    if not isinstance(raw, dict):
        return out
    for value in raw.values():
        if isinstance(value, dict) and "angle" in value and "distance" in value:
            out.append({"angle": float(value["angle"]), "distance": float(value["distance"])})
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            out.append({"angle": float(value[0]), "distance": float(value[1])})
    return out


def norm_pi(angle: float) -> float:
    """Normalize to [-pi, pi) — counter-clockwise positive (left)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def direction_of(angle: float) -> str:
    """Quadrant label of a counter-clockwise angle: Front within 45 degrees, then Left, Back, Right."""
    a = angle % (2 * math.pi)
    if a <= math.pi / 4 or a >= 7 * math.pi / 4:
        return "Front"
    if a <= 3 * math.pi / 4:
        return "Left"
    if a <= 5 * math.pi / 4:
        return "Back"
    return "Right"


def action_options(cands: list[dict[str, float]]) -> dict[str, list[int]]:
    options: dict[str, list[int]] = {"Left": [], "Front": [], "Right": [], "Back": []}
    for i, cand in enumerate(cands):
        options[direction_of(cand["angle"])].append(i + 1)
    return options


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


def annotate_strip(
    views: list[dict[str, Any]], cands: list[dict[str, float]], view_px: int
) -> bytes:
    """[Left|Front|Right|Back] hstack with numbered waypoint markers.

    Angle→x mapping: for counter-clockwise angle θ normalized to [-pi, pi),
    ``x = w_single * (1.5 - 2θ/π)`` — Front centers on tile 2, Left on tile 1,
    Right on tile 3, Back wraps across the strip ends onto tile 4.
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw

    by_dir = {v.get("dir_id"): v for v in views}
    tiles: list[Any] = []
    for dir_id, _label in STRIP_DIRS:
        view = by_dir.get(dir_id)
        if view is None or not view.get("rgb_base64"):
            raise ValueError(f"panorama view dir_id={dir_id} missing")
        tiles.append(PILImage.open(BytesIO(base64.b64decode(view["rgb_base64"]))).convert("RGB"))

    w_single, h = tiles[0].size
    w_full = w_single * 4
    label_h = max(20, h // 14)
    canvas = PILImage.new("RGB", (w_full, h + label_h), (255, 255, 255))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * w_single, label_h))

    draw = ImageDraw.Draw(canvas)
    label_font = _font(int(label_h * 0.7))
    for i, (_dir_id, label) in enumerate(STRIP_DIRS):
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (
                (i + 0.5) * w_single - (bbox[2] - bbox[0]) / 2,
                (label_h - (bbox[3] - bbox[1])) / 2 - bbox[1],
            ),
            label,
            fill=(0, 0, 0),
            font=label_font,
        )

    radius = max(12, h // 20)
    num_font = _font(int(radius * 1.15))
    y = label_h + h // 2
    for i, cand in enumerate(cands):
        theta = norm_pi(cand["angle"])
        x = w_single * (1.5 - 2 * theta / math.pi)
        if x < 0:
            x += w_full
        x = min(max(x, radius), w_full - radius)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(0, 255, 0),
            outline=(0, 128, 0),
            width=2,
        )
        text = str(i + 1)
        bbox = draw.textbbox((0, 0), text, font=num_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2 - bbox[0], y - (bbox[3] - bbox[1]) / 2 - bbox[1]),
            text,
            fill=(255, 0, 0),
            font=num_font,
        )

    if view_px and h > view_px:  # view_px=0 → native resolution, no downscale
        scale = view_px / h
        canvas = canvas.resize(
            (int(canvas.width * scale), int(canvas.height * scale)), PILImage.LANCZOS
        )
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def render_and_predict(
    s: Session, *, remaining: bool, empty_note: str, log_key: str
) -> tuple[bytes, dict[str, Any]]:
    """Render the RGB-D panorama, predict waypoints, annotate the strip, and set the
    candidate list to the fresh numbering. Pure read — advances the frame counter,
    never the simulator or the step budget. ``remaining`` adds steps_remaining_approx
    to the status, ``empty_note`` is the note when nothing was predicted, ``log_key``
    names the look in the live log — the two looks word these their own way."""
    pano = s.call("panorama", timeout=ENV_TIMEOUT_S, n_views=N_PANO_VIEWS)
    if "error" in pano:
        raise RuntimeError(f"panorama: {pano['error']}")
    views = pano.get("views") or []
    # forward only what the predictor reads — drops depth_raw_base64 (~4 MB)
    slim = [
        {
            "dir_id": v.get("dir_id"),
            "rgb_base64": v.get("rgb_base64"),
            "depth_base64": v.get("depth_base64"),
        }
        for v in views
    ]
    predict_fn = s.options.get("wp_predict_fn") or "smartway_waypoint__predict"
    pred = call_wp(s, predict_fn, {"views": slim})
    cands = normalize_candidates(pred.get("candidates"))
    s.notes["wp_candidates"] = cands

    view_px = s.options.get("wp_view_px")
    png = annotate_strip(views, cands, 384 if view_px is None else int(view_px))
    s.obs_count += 1
    s.live_write(f"obs_{s.obs_count:04d}_step{s.steps_taken:03d}.png", png)

    steps: dict[str, Any] = {"steps_taken_total": s.steps_taken}
    if remaining:
        steps["steps_remaining_approx"] = max(0, s.step_budget - s.steps_taken)
    status: dict[str, Any] = {
        "waypoints": {
            str(i + 1): {
                "direction": direction_of(c["angle"]),
                "angle_deg": round(math.degrees(norm_pi(c["angle"])), 1),
                "distance_m": round(c["distance"], 2),
            }
            for i, c in enumerate(cands)
        },
        "action_options": action_options(cands),
        **steps,
        **move_fields(s),
        **budget_fields(s),
    }
    if not cands:
        status["note"] = empty_note
    s.live_log({log_key: True, "num_waypoints": len(cands)})
    return png, status
