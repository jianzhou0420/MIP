"""SlamHook — SLAM as a hook on EmbodiedScoreEnv (mip/envs/embodiedscore/hook.py), in the env
server's process: the SLAM middleware over the benchmark line, every frame fed to ORB-SLAM3,
and the instruments' reads served as env verbs.

    env:
      _target_: mip.envs.embodiedscore.env.EmbodiedScoreEnv
      env_fields: [instruments]        # the tool set's switch, pushed with configure()
      hooks:
      - _target_: mip.tools.slam.hook.SlamHook
        slam: {cell_size_m: 0.1, …}    # world.stack.SLAM_KEYS

- the stack: ``world.stack.build`` — the STANDARD body at micro quanta (0.05 m / 3° / 6°,
  RGB-aligned metric depth) under the line's metric wrapper and a coarse-step TimeLimit; each
  coarse action is 5 fed micro-steps, a blocked forward aborts and marks the map; the 360°
  bootstrap scan at reset when instruments are on. ``instruments``: 0 off, 1 SLAM + map v1,
  2 SLAM + map v2.
- step info: collision (a forward that did not move the body) · collisions · micro_steps_done ·
  slam · reanchors · tick (· pitch_deg / tilt_clamped on a LOOK); metrics: steps_taken · stop_called ·
  collisions · instruments, the question lines' letter score, and with instruments the
  SLAM-quality block (ATE, tracking ratio, map epochs …).
- verbs: the six instruments get_pose · get_trajectory · get_map · look_around · mark ·
  preview_path (SLAM estimates, never ground truth; refused below instruments 1), and the
  waypoint tools' two verbs on the SLAM body — step_hightolow turns to the heading as fed
  micro-turns (a tracker cannot follow a pose write) and walks charged coarse FORWARDs;
  panorama renders in place, SLAM-silent, depth at the predictor's 256 px.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from mip.envs.embodiedscore.hook import EnvHook

_INSTRUMENT_LEVELS = (0, 1, 2)
_ACTION_STOP = 0
_LETTERS = "ABCD"
# Pose/map frame facts: world anchored at the episode start pose, up on the map = +z =
# starting heading, right = +x, yaw_deg increases turning RIGHT, 0 = starting heading.
_POSE_CONVENTION = (
    "x right / z forward of your START pose (metres); "
    "yaw_deg: 0 = your starting heading, increases turning "
    "right; on the map, up = +z (your starting heading)"
)


class SlamHook(EnvHook):
    fields = ("instruments",)

    def __init__(self, slam: dict[str, Any] | None = None) -> None:
        from mip.tools.slam.world import stack

        self._slam: dict[str, Any] = dict(slam or {})
        unknown = set(self._slam) - stack.SLAM_KEYS
        if unknown:
            raise ValueError(
                f"env.slam: unknown key(s) {sorted(unknown)} (known: {sorted(stack.SLAM_KEYS)})"
            )
        self._instruments = 0

    # ── the hook contract ──

    def configure(self, **fields: Any) -> None:
        if "instruments" in fields:
            level = int(fields["instruments"])
            if level not in _INSTRUMENT_LEVELS:
                raise ValueError(f"instruments must be one of {_INSTRUMENT_LEVELS}, got {level}")
            self._instruments = level

    def make_stack(
        self, line: str, split: str, *, data_root: str, scene_root: str, gpu_id: int
    ) -> Any:
        from mip.tools.slam.world import stack

        return stack.build(
            line,
            split,
            instruments=self._instruments,
            data_root=data_root,
            scene_root=scene_root,
            gpu_id=gpu_id,
            slam=self._slam,
        )

    def step_info(self, info: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "collision": bool(info.get("collision", False)),
            "collisions": int(info.get("collisions", 0)),
            "micro_steps_done": int(info.get("micro_steps_done", 0)),
            "slam": str(info.get("slam", "")),
            "reanchors": int(info.get("reanchors", 0)),
            "tick": int(info.get("tick", 0)),
        }
        if "pitch_deg" in info:
            # the sensor read carries float32 noise (-59.9999992); rounding reports exact multiples
            out["pitch_deg"] = round(float(info["pitch_deg"]), 3)
            if info.get("tilt_clamped"):
                out["tilt_clamped"] = True
        return out

    def metrics(self, metrics: dict[str, Any]) -> None:
        """The tool set's bookkeeping and — instrumented — the SLAM-quality block. On a question
        line ``success`` is the answer letter against the dataset's (unanswered = 0)."""
        mgr, se = self.mgr, self._se
        metrics.update(
            {
                "steps_taken": float(mgr.step_count),
                "stop_called": float(self._stop_called()),
                "collisions": float(se.env.collisions) if se is not None else 0.0,
                "instruments": float(self._instruments),
            }
        )
        episode = mgr.base.episode
        if episode.goal.kind == "question":
            metrics.update(_score_answer(mgr.answer or "", str(episode.goal.answer)))
            metrics["label"] = str(episode.info.get("label", ""))
            metrics["final_pitch_deg"] = (
                round(float(se.env.pitch_deg), 3) if se is not None else 0.0
            )
        if self._instruments and se is not None:
            slam = se.metrics()
            ate = slam.get("ate_rmse_m")
            metrics.update(
                {
                    "ate_rmse_m": None if ate is None or math.isnan(ate) else round(float(ate), 4),
                    "tracking_ratio": round(float(slam["tracking_ratio"]), 4),
                    "lost_ticks": float(slam["lost_ticks"]),
                    "reanchors": float(slam["reanchors"]),
                    # The atlas, as three numbers: how many maps ORB-SLAM3
                    # ended up with, how many it reconnected, and how many are
                    # still sitting on a guessed anchor nobody ever checked.
                    "map_epochs": float(len(slam["epochs"])),
                    "map_merges": float(len(slam["merges"])),
                    "map_provisional": float(sum(e["provisional"] for e in slam["epochs"])),
                    "explored_m2": round(float(slam["explored_m2"]), 2),
                    "micro_ticks": float(slam["ticks"]),
                }
            )

    def verbs(self) -> dict[str, Callable[..., dict]]:
        return {
            "get_pose": self.get_pose,
            "get_trajectory": self.get_trajectory,
            "get_map": self.get_map,
            "look_around": self.look_around,
            "mark": self.mark,
            "preview_path": self.preview_path,
            "step_hightolow": self.step_hightolow,
            "panorama": self.panorama,
        }

    # ── the episode's SLAM session ──

    @property
    def _micro(self) -> Any:
        from mip.tools.slam.world import stack

        e = self.mgr.stack
        while e is not None:
            if isinstance(e, stack.SlamMicroStep):
                return e
            e = getattr(e, "env", None)
        raise RuntimeError("SlamMicroStep not found in the stack")

    @property
    def _se(self) -> Any:
        return None if self.mgr.stack is None else self._micro.slam

    @property
    def _render(self) -> dict[str, Any]:
        """render_annotated_map(_v2) kwargs from env.slam.render (out_px /
        margin_m / min_window_m / grid_every_m); {} = the renderer's defaults."""
        return dict(self._slam.get("render") or {})

    def _stop_called(self) -> bool:
        return bool(self.mgr.current_info.get("stop_called", False))

    def _instrumented_unlocked(self, tool: str) -> dict | None:
        if not self.mgr.initialized or self._se is None or not self.mgr.placed:
            return {"error": "no live episode — set an episode first"}
        if not self._instruments:
            return {"error": f"{tool} requires the SLAM instruments (instruments >= 1)"}
        return None

    # ── the instruments (free; SLAM estimates, not ground truth) ──

    def get_pose(self) -> dict:
        with self.mgr.lock:
            err = self._instrumented_unlocked("get_pose")
            if err:
                return {"pose": err}
            pose = self._se.get_pose()
            pose["convention"] = _POSE_CONVENTION
            return {"pose": pose}

    def get_trajectory(self) -> dict:
        with self.mgr.lock:
            err = self._instrumented_unlocked("get_trajectory")
            return {"trajectory": err or self._se.get_trajectory()}

    def get_map(self, floor: int | None = None, view: str = "global") -> dict:
        result = self._get_map(
            int(floor) if floor not in (None, "") else None, str(view or "global")
        )
        if "error" in result:
            return {"map": None, "map_local": None, "frontiers": [], "info": result}
        frontiers = result.pop("frontiers", [])
        out = _image_fields(
            {"map": result.pop("map", None), "map_local": result.pop("map_local", None)},
            ("map", "map_local"),
        )
        out.update(frontiers=frontiers, info=result)
        return out

    def _get_map(self, floor: int | None, view: str) -> dict:
        from mip.tools.slam.world._frontier import frontier_mask
        from mip.tools.slam.world._map_render import render_annotated_map

        with self.mgr.lock:
            err = self._instrumented_unlocked("get_map")
            if err:
                return err
            se = self._se
            # v3 is v2's floor registry with evidence layers underneath:
            # same layers, same render window, same picture.
            if se.map_mode in ("v2", "v3"):
                return self._get_map_v2_unlocked(se, floor=floor, view=view)
            m = se.get_map()
            vm = se.view()
            m.pop("png", None)
            frontier_cells = [(int(f["id"][1:]), tuple(f.pop("cell"))) for f in m["frontiers"]]
            agent_xz = None
            yaw = None
            if se.last_pose is not None:
                agent_xz = (float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
                yaw = math.radians(se.get_pose()["yaw_deg"])
            img, window = render_annotated_map(
                vm.grid,
                frontier_mask(vm.grid),
                vm.cell_size,
                vm.origin,
                agent_xz=agent_xz,
                agent_yaw=yaw,
                trajectory=se.drawn_track(),
                frontier_cells=frontier_cells,
                **self._render,
            )
            m["map"] = img
            p = se.get_pose()
            if "x" in p:
                m["pose"] = {"x": p["x"], "z": p["z"], "yaw_deg": p["yaw_deg"]}
            m["map_window_m"] = round(window, 1)
            m["orientation"] = "up = +z (your starting heading); right = +x"
            return m

    def _get_map_v2_unlocked(
        self,
        se: Any,
        floor: int | None = None,
        view: str = "global",
        path: list | None = None,
        store_window: bool = True,
    ) -> dict:
        """Map v2: no frontier layer; per-floor render; crop window
        snapped to 2 m multiples and grow-only per episode; ``floor`` views
        another storey; ``view="local"`` adds a fixed 8 m agent-centred crop;
        ``path`` overlays a preview polyline; markers render on their floor."""
        from mip.tools.slam.world._map_render import render_annotated_map_v2

        vm = se.view()
        n_floors = len(vm.layers)
        if view == "local" or floor is None:
            fidx = vm.current_floor
        else:
            fidx = int(floor) - 1
            if not 0 <= fidx < n_floors:
                return {"error": f"floor {floor} out of range (1..{n_floors})"}
        layer = vm.layers[fidx]
        floor_y = vm.floor_ys[fidx]
        on_floor = fidx == vm.current_floor
        agent_xz = None
        yaw = None
        if se.last_pose is not None and on_floor:
            agent_xz = (float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
            yaw = math.radians(se.get_pose()["yaw_deg"])
        # this sub-goal's path, inside this episode's — see SlamEnv.drawn_track
        traj = [p for p in se.drawn_track() if abs(p[1] - floor_y) <= vm.floor_merge_m]
        floor_label = f"floor {fidx + 1}/{n_floors}"
        # where this EPISODE began, independent of how much of the path is drawn —
        # the slice above would otherwise hide the start marker from sub-goal 2 on
        start_here = (
            len(se.est_track) > se.ep_track0
            and abs(se.est_track[se.ep_track0][1] - floor_y) <= vm.floor_merge_m
        )
        markers = [(m["label"], m["x"], m["z"]) for m in se.markers if m["floor"] == fidx]
        img, window, new_window = render_annotated_map_v2(
            layer.grid,
            vm.cell_size,
            vm.origin,
            agent_xz=agent_xz,
            agent_yaw=yaw,
            trajectory=traj,
            window_cells=se.map_window,
            floor_label=floor_label,
            mark_start=start_here,
            markers=markers,
            path=path,
            **self._render,
        )
        if store_window:
            se.map_window = new_window
        out = {
            "map": img,
            "floor": floor_label,
            "view": view,
            "free_m2": layer.free_area_m2(),
            "explored_m2": layer.explored_area_m2(),
            "markers": [{"label": lb, "x": mx, "z": mz} for lb, mx, mz in markers],
            "slam": se.status,
            "tick": se.tick,
            "map_window_m": round(window, 1),
            "orientation": "up = +z (your starting heading); right = +x",
        }
        if view == "local":
            img_l, window_l, _ = render_annotated_map_v2(
                layer.grid,
                vm.cell_size,
                vm.origin,
                agent_xz=agent_xz,
                agent_yaw=yaw,
                trajectory=traj,
                floor_label=floor_label,
                mark_start=start_here,
                markers=markers,
                path=path,
                fixed_window_cells=_local_window_cells(se),
                **self._render,
            )
            out["map_local"] = img_l
            out["local_window_m"] = round(window_l, 1)
        p = se.get_pose()
        if "x" in p:
            out["pose"] = {"x": p["x"], "z": p["z"], "yaw_deg": p["yaw_deg"]}
        return out

    def look_around(self) -> dict:
        """360° panorama (4 views 90° apart), SLAM-silent, free. On the
        question lines a step-cap truncation only ends MOVEMENT — the look
        stays available until the answer."""
        with self.mgr.lock:
            mgr = self.mgr
            result = self._instrumented_unlocked("look_around")
            if result is None:
                question = mgr.base.episode.goal.kind == "question"
                if mgr.episode_done and not (question and not self._stop_called()):
                    result = {"error": "episode already done"}
                else:
                    result = self._se.look_around()
                    result["pitch_deg"] = round(float(result.get("pitch_deg", 0.0)), 3)
                    result["step_count"] = mgr.step_count
        if "error" in result:
            return {
                "view_front": None,
                "view_right": None,
                "view_back": None,
                "view_left": None,
                "info": result,
            }
        views = result.pop("views")
        out = {
            "view_front": views[0],
            "view_right": views[1],
            "view_back": views[2],
            "view_left": views[3],
        }
        out = _image_fields(out, ("view_front", "view_right", "view_back", "view_left"))
        out["info"] = result
        return out

    def mark(self, label: str) -> dict:
        with self.mgr.lock:
            err = self._instrumented_unlocked("mark")
            return {"result": err or self._se.mark(str(label))}

    def preview_path(self, x: float, z: float) -> dict:
        with self.mgr.lock:
            result = self._instrumented_unlocked("preview_path")
            if result is None:
                se = self._se
                plan = _plan(se, float(x), float(z))
                if "error" in plan or not plan.get("reachable"):
                    result = plan
                else:
                    result = self._get_map_v2_unlocked(se, path=plan["world"], store_window=False)
                    result.update(
                        {
                            "reachable": True,
                            "distance_m": plan["distance_m"],
                            "snapped_target": plan["snapped_target"],
                            "snapped": plan["snapped"],
                        }
                    )
        if "error" in result:
            return {"map": None, "result": result}
        out = _image_fields({"map": result.pop("map", None)}, ("map",))
        out["result"] = result
        return out

    # ── the waypoint tools' verbs on the SLAM body ──

    def step_hightolow(self, angle: float, distance: float) -> dict:
        """The waypoint move under SLAM. Turn toward the waypoint by ``angle`` (rad,
        counter-clockwise positive) as fed micro-turns (``SlamEnv.turn_exact``: zero
        coarse steps, every frame reaches ORB-SLAM3 and the final heading is exact), then
        walk ``distance`` (m) as ``int(distance // 0.25)`` coarse FORWARDs through the whole
        stack (5 fed micro-steps each, charged to the budget and the metrics; a blocked
        forward is marked on the map). A blocked forward does not end the walk, as in the
        plain waypoint move. ``angle == 0 and distance == 0`` dispatches STOP. Returns ``step()``'s shape; the
        move's accounting rides in ``info`` (``step_count`` is what goto syncs to). No
        ground-truth pose is read anywhere on the move."""
        from mip.tools.slam.world import stack

        mgr = self.mgr
        if float(angle) == 0.0 and float(distance) == 0.0:
            return mgr.step(_ACTION_STOP)
        with mgr.lock:
            if not mgr.initialized:
                return {"error": "Environment not initialized"}
            mgr.ensure_placed()
            if mgr.episode_done:
                return {"error": "Episode already done", "terminated": True}
            se = self._se
            if se is None:
                return {"error": "no SLAM session -- set an episode first"}
            turn = se.turn_exact(float(angle))
            # the turn leaves no gym frame behind: refresh the cached view so a later
            # observe() shows the post-turn heading, and the SLAM facts
            mgr.refresh_view(se.env.observe(), self._micro._slam_facts())
            forward_step = float(stack.COARSE_STEP_M)
            # int(distance // step), NOT max(1, ...): a rotate-only request walks ZERO forwards
            ksteps = int(float(distance) // forward_step)
            flags = {"terminated": False, "truncated": False}
            moved_m, collision_count, last_collided, last_delta = 0.0, 0, False, 0.0
            for _ in range(ksteps):
                if mgr.episode_done:
                    break
                p0 = np.asarray(mgr.base.world.pose()[0], dtype=np.float64)
                flags = mgr.advance(1)  # FORWARD
                p1 = np.asarray(mgr.base.world.pose()[0], dtype=np.float64)
                last_delta = float(np.linalg.norm((p1 - p0)[[0, 2]]))
                moved_m += last_delta
                last_collided = bool(mgr.current_info.get("collision", False))
                if last_collided:
                    collision_count += 1
            info = mgr.step_info(
                angle=float(angle),
                distance=float(distance),
                ksteps=ksteps,
                micro_turns=int(turn["micro_turns"]),
                requested_distance_m=round(ksteps * forward_step, 3),
                actual_translation_m=round(moved_m, 3),
                collision_count=collision_count,
                stopped_at_collision=bool(last_collided and last_delta < 0.05),
            )
            return {"reward": 0.0, **flags, "info": info}

    def panorama(self, n_views: int = 12) -> dict:
        """Aligned RGB + depth views at ``n_views`` headings from the current pose,
        rendered in place through ``render_at``: the body never moves, nothing is fed
        (SLAM-silent, like look_around), nothing is charged. View i faces i * 360/n deg
        counter-clockwise (the waypoint strip's dir_id convention)."""
        import quaternion  # enables np.quaternion

        mgr = self.mgr
        with mgr.lock:
            if not mgr.initialized:
                return {"error": "Environment not initialized"}
            mgr.ensure_placed()
            world = mgr.base.world
            pos, rot = world.pose()
            q = np.quaternion(rot[3], rot[0], rot[1], rot[2])
            n = int(n_views)
            angle_step = 2.0 * np.pi / n
            views = []
            for i in range(n):
                yaw = i * angle_step
                r = q * np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
                obs = world.render_at(pos, [r.x, r.y, r.z, r.w])
                view: dict[str, Any] = {
                    "dir_id": i,
                    "heading_deg": round(float(np.degrees(yaw)) % 360, 1),
                    "rgb_base64": _encode_rgb_base64(obs["rgb"]),
                }
                if "depth" in obs:
                    depth_m = _predictor_depth(np.squeeze(obs["depth"]))
                    view["depth_base64"] = _encode_depth_base64(depth_m)
                    view["depth_raw_base64"] = _encode_depth_raw_base64(depth_m)
                views.append(view)
            return {"views": views, "n_views": len(views)}


def _score_answer(pred_letter: str, gt_letter: str) -> dict:
    """The multiple-choice score: pred == gt -> success 1.0; an empty / invalid
    prediction is the benchmark's own zero (an episode that never answered)."""
    pred = str(pred_letter or "").strip().strip("\"'.()").upper()
    gt = str(gt_letter or "").strip().upper()
    answered = len(pred) == 1 and pred in _LETTERS
    return {
        "success": float(answered and pred == gt),
        "answered": float(answered),
        "pred_letter": pred if answered else "",
        "gt_letter": gt,
    }


def _local_window_cells(se: Any, span_m: float = 8.0) -> tuple:
    vm = se.view()
    cs = vm.cell_size
    n = vm.n
    origin = vm.origin
    snap = max(1, round(2.0 / cs))
    side = -(-round(span_m / cs) // snap) * snap
    ax = az = 0.0
    if se.last_pose is not None:
        ax, az = float(se.last_pose[0, 3]), float(se.last_pose[2, 3])
    azi = math.floor(az / cs) + origin
    axi = math.floor(ax / cs) + origin
    zi0 = origin + ((azi - side // 2 - origin) // snap) * snap
    xi0 = origin + ((axi - side // 2 - origin) // snap) * snap
    zi0 = max(0, min(zi0, n - side))
    xi0 = max(0, min(xi0, n - side))
    return (zi0, zi0 + side, xi0, xi0 + side)


def _plan(se: Any, x: float, z: float, goal_snap_m: float = 1.0) -> dict:
    from mip.tools.slam.world._plan import astar_free, nearest_free, path_length_m

    vm = se.view()
    if se.last_pose is None:
        return {"error": "no pose estimate yet"}
    grid = vm.grid
    cs = vm.cell_size
    start = vm.world_to_cell(float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
    goal = vm.world_to_cell(float(x), float(z))
    s2 = nearest_free(grid, start, max(1, int(0.5 / cs)))
    if s2 is None:
        return {"error": "no free map cells near the agent yet"}
    g2 = nearest_free(grid, goal, max(1, int(goal_snap_m / cs)))
    if g2 is None:
        return {
            "reachable": False,
            "reason": (
                f"target has no walked-free map cell within {goal_snap_m:g} m — "
                "the map only knows space you have already seen"
            ),
        }
    cells = astar_free(grid, s2, g2)
    if cells is None:
        return {"reachable": False, "reason": "no free-space route on this floor's map"}
    world = [vm.cell_to_world(zi, xi) for zi, xi in cells]
    return {
        "reachable": True,
        "cells": cells,
        "world": world,
        "distance_m": round(path_length_m(cells, cs), 2),
        "snapped_target": {"x": round(world[-1][0], 2), "z": round(world[-1][1], 2)},
        "snapped": g2 != goal,
    }


def _image_fields(out: dict, keys: tuple[str, ...]) -> dict:
    """uint8 images → base64 PNG on the wire (what the bridge decodes)."""
    from mip.core.envserver import png_b64

    for k in keys:
        v = out.get(k)
        if v is not None and not isinstance(v, str):
            out[k] = png_b64(np.asarray(v, dtype=np.uint8))
    return out


def _predictor_depth(depth_m: np.ndarray) -> np.ndarray:
    """The panorama depth at the STANDARD body's depth spec (256 px), which is what
    the waypoint predictor's depth encoder takes. The SLAM body renders depth aligned
    with RGB (512 px: ORB-SLAM3's RGB-D pair), so block-average it down (exact integer
    ratio; bilinear otherwise) — the one place the predictor inputs differ from the
    plain waypoint set's."""
    from mip.tools.slam.world import stack

    spec = stack.bodies.STANDARD.depth
    th, tw = int(spec.height), int(spec.width)
    d = np.asarray(depth_m, dtype=np.float32)
    h, w = d.shape[:2]
    if (h, w) == (th, tw):
        return d
    if h % th == 0 and w % tw == 0:
        fh, fw = h // th, w // tw
        return d[: th * fh, : tw * fw].reshape(th, fh, tw, fw).mean(axis=(1, 3)).astype(np.float32)
    from PIL import Image

    return np.asarray(
        Image.fromarray(d, mode="F").resize((tw, th), Image.BILINEAR), dtype=np.float32
    )


def _encode_rgb_base64(rgb: np.ndarray) -> str:
    """The waypoint panorama's RGB encoding (PNG, base64)."""
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb)[:, :, :3].astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_depth_base64(depth: np.ndarray) -> str:
    """The normalized 8-bit depth preview (what the predictor reads)."""
    from PIL import Image

    d = np.squeeze(depth)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min > 1e-6:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(d_norm, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_depth_raw_base64(depth_m: np.ndarray) -> str:
    """16-bit PNG depth in millimetres (absolute metric depth)."""
    from PIL import Image

    d = np.squeeze(depth_m).astype(np.float32)
    d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(d_mm, mode="I;16").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
