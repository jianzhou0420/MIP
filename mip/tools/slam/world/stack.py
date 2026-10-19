"""stack — the SLAM environment stack on EmbodiedScore-envs.

    base    = build_env(line, split, body=FINE, actions=STANDARD)   # the package body at micro quanta
    slam    = SlamMicroStep(base, instruments=...)                   # coarse action -> MICRO_PER_ACTION fine steps, each fed
    metrics = <the line's metric wrapper>(slam)                      # counts COARSE steps, as the plain env does
    env     = TimeLimit(metrics, 500)                                # coarse budget

FINE is the STANDARD preset with its three quanta divided by
``MICRO_PER_ACTION`` (0.25 m -> 0.05 m, 15° -> 3°, 30° -> 6°) and a depth
camera at the RGB resolution (ORB-SLAM3 RGBD wants an aligned pair); depth
stays METRIC (no DepthClip — the SLAM feed and the agent's frame are the
same array). The clamp (±60°), agent cylinder, sliding, navmesh file and
camera mast are the STANDARD body's, so a coarse action lands exactly
where the plain STANDARD step would (turns compose exactly; a forward is
five collinear 0.05 m tries).

What the wrapper adds to ``info`` per coarse step: ``action``,
``micro_steps_done``, ``collision`` (a forward that did not move — the
remaining micro-steps are aborted and the map marks an obstacle ahead),
``collisions`` (episode count), ``slam`` (tracking status),
``reanchors``, ``tick`` (micro clock), ``pitch_deg`` and ``tilt_clamped`` on
a LOOK action. STOP / SUBTASK_STOP pass through as ONE base step.

The SLAM session (``SlamEnv``) is rebuilt at every reset — the occupancy
grid and the ORB-SLAM3 anchor belong to the episode — and bootstrapped with
the disclosed 360° scan when instruments are on. The simulator itself is
the package's and persists across episodes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import embodiedscore_envs as es
import gymnasium as gym
from embodiedscore_envs.benchmarks import build_env
from embodiedscore_envs.benchmarks.env import Act, CameraSpec
from embodiedscore_envs.benchmarks.presets import actions, bodies
from gymnasium.wrappers import TimeLimit

from ._body import WorldBody
from ._slam_env import MICRO_PER_ACTION, SlamEnv

COARSE_STEP_M = bodies.STANDARD.forward_step_m       # 0.25
COARSE_TURN_DEG = bodies.STANDARD.turn_deg           # 15
COARSE_TILT_DEG = bodies.STANDARD.tilt_deg           # 30
PITCH_LIMIT_DEG = bodies.STANDARD.tilt_limit_deg     # 60
MAX_COARSE_STEPS = 500
CELL_SIZE_M = 0.10
MAP_SIZE_M = 48.0

FINE = dataclasses.replace(
    bodies.STANDARD,
    forward_step_m=COARSE_STEP_M / MICRO_PER_ACTION,
    turn_deg=COARSE_TURN_DEG / MICRO_PER_ACTION,
    tilt_deg=COARSE_TILT_DEG / MICRO_PER_ACTION,
    depth=CameraSpec(bodies.STANDARD.rgb.width, bodies.STANDARD.rgb.height,
                     bodies.STANDARD.rgb.hfov_deg, bodies.STANDARD.rgb.position),
)


# the env block's ``slam`` keys (an experiment's env.slam): per-experiment overrides of
# the module constants above and of SlamEnv's defaults
SLAM_KEYS = frozenset({
    "cell_size_m", "map_size_m", "max_coarse_steps", "frontier_match_m",
    "start_pitch_deg", "map_mode", "mapper", "render",
})


class SlamMicroStep(gym.Wrapper):
    """Coarse actions in, MICRO_PER_ACTION fine steps + auto-fed SLAM inside."""

    def __init__(self, env: gym.Env, instruments: int = 0, slam: dict[str, Any] | None = None) -> None:
        super().__init__(env)
        self.instruments = int(instruments)
        self.slam_cfg: dict[str, Any] = dict(slam or {})
        unknown = set(self.slam_cfg) - SLAM_KEYS
        if unknown:
            raise ValueError(f"env.slam: unknown key(s) {sorted(unknown)} (known: {sorted(SLAM_KEYS)})")
        self.slam: SlamEnv | None = None
        self.boot: dict | None = None

    def _slam_kwargs(self, base: Any) -> dict[str, Any]:
        """SlamEnv's kwargs: the stack's quanta + the env block's ``slam``
        keys. An absent key = the constant it replaces (cell_size_m /
        map_size_m) or SlamEnv's own default (frontier_match_m /
        start_pitch_deg / mapper); ``map_mode`` null = v2 at instruments 2,
        v1 otherwise. ``v3`` is v2's floor layers over evidence rather than
        verdicts — set per experiment file, never by default."""
        s = self.slam_cfg
        kw: dict[str, Any] = dict(
            coarse_step_m=COARSE_STEP_M, coarse_turn_deg=COARSE_TURN_DEG,
            cell_size=float(s.get("cell_size_m") or CELL_SIZE_M),
            map_size_m=float(s.get("map_size_m") or MAP_SIZE_M),
            pose_source="slam" if self.instruments else "gt",
            map_mode=str(s.get("map_mode") or ("v2" if self.instruments == 2 else "v1")),
            coarse_tilt_deg=COARSE_TILT_DEG if base.body.has_tilt else None,
            pitch_limit_deg=PITCH_LIMIT_DEG,
        )
        for key in ("frontier_match_m", "start_pitch_deg"):
            if s.get(key) is not None:
                kw[key] = float(s[key])
        if s.get("mapper"):
            kw["mapper"] = dict(s["mapper"])
        return kw

    # ── episode ──

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        if self.slam is not None:
            self.slam.close()          # ends the ORB-SLAM3 container; the simulator stays
        base = self.env.unwrapped
        self.slam = SlamEnv(WorldBody(base), **self._slam_kwargs(base))
        self.slam.reset()
        self.boot = self.slam.bootstrap() if self.instruments else None
        info = dict(info)
        info.update(self._slam_facts())
        if self.boot is not None:
            info["bootstrap"] = dict(self.boot)
        return obs, info

    def close(self) -> None:
        if self.slam is not None:
            self.slam.close()
            self.slam = None
        self.env.close()

    # ── the only way the world moves ──

    def step(self, action: int):
        a = int(action)
        se = self.slam
        if se is None:
            raise RuntimeError("SlamMicroStep: reset() first")
        if a in (Act.STOP, Act.SUBTASK_STOP):
            obs, reward, terminated, truncated, info = self.env.step(a)
            info = dict(info)
            info.update({"action": a, "micro_steps_done": 0, "collision": False})
            info.update(self._slam_facts())
            return obs, reward, terminated, truncated, info
        # a movement or LOOK action: SlamEnv drives the micro-steps through
        # WorldBody.step == HabitatEnv.step on the FINE body, feeding SLAM
        # after each; the last micro-step's gym tuple is what goes outward
        act = se.act(a)
        if se.env.last is None:        # only a LOOK clamped to a no-op before any other step could get here
            obs, reward, terminated, truncated, info = self.env.step(a)
        else:
            # a clamped LOOK that did nothing (micro_steps_done 0) is charged
            # like any coarse action and re-emits the previous frame/facts
            obs, reward, terminated, truncated, info = se.env.last
        info = dict(info)
        info.update({
            "action": a,
            "micro_steps_done": int(act.get("micro_steps_done", 0)),
            "collision": bool(act.get("collision", False)),
        })
        if "pitch_deg" in act:
            info["pitch_deg"] = float(act["pitch_deg"])
            info["tilt_clamped"] = bool(act.get("tilt_clamped", False))
        info.update(self._slam_facts())
        return obs, reward, terminated, truncated, info

    def _slam_facts(self) -> dict:
        se = self.slam
        if se is None:
            return {}
        return {"slam": se.status, "reanchors": int(se.reanchors), "tick": int(se.tick),
                "collisions": int(se.env.collisions)}


def build(line: str, split: str, *, instruments: int = 0, data_root: str | None = None,
          scene_root: str | None = None, gpu_id: int = 0, max_coarse_steps: int | None = None,
          slam: dict[str, Any] | None = None) -> gym.Env:
    """The SLAM stack for a benchmark line (standard task semantics,
    FINE body): TimeLimit(<line metrics>(SlamMicroStep(build_env(...)))).
    ``slam`` = the env block's ``slam`` dict (SLAM_KEYS); ``max_coarse_steps``
    None = its ``max_coarse_steps`` or MAX_COARSE_STEPS."""
    slam = dict(slam or {})
    if max_coarse_steps is None:
        max_coarse_steps = int(slam.get("max_coarse_steps") or MAX_COARSE_STEPS)
    b = es.benchmark(line, "standard")
    base = build_env(b.name, split, data_root=data_root, scene_root=scene_root, body=FINE,
                     actions=actions.STANDARD, gpu_id=gpu_id)
    env: gym.Env = SlamMicroStep(base, instruments=instruments, slam=slam)
    if b.metrics is not None:
        env = b.metrics(env)
    return TimeLimit(env, max_episode_steps=int(max_coarse_steps))


__all__ = [
    "COARSE_STEP_M",
    "COARSE_TILT_DEG",
    "COARSE_TURN_DEG",
    "FINE",
    "MAX_COARSE_STEPS",
    "PITCH_LIMIT_DEG",
    "SLAM_KEYS",
    "SlamMicroStep",
    "build",
]
