"""EmbodiedScoreEnv — one EmbodiedScore-envs stack behind a verb surface.

The ``env`` layer of an experiment file, instantiated by ``core.envserver`` in
the interpreter the file names (``env.python``) and reached by the runner
and the arm's bridge over HTTP (``core.envclient``): every public method of
``EmbodiedScoreEnv`` is one verb, ``POST /<method>`` with its kwargs as JSON.

    env:
      _target_: exp_workspace.bareES.mcp.env.EmbodiedScoreEnv
      variant: standard          # the shared STANDARD body, or the line's own upstream rig
      python: ac-es              # interpreter (conda env name or path) the server runs under
      port: 9200

The line and split are the TASK's (``task.dataset`` / ``task.split``): the
runner pushes them with ``configure`` before the first placement, so one
server serves whichever line the experiment measures.

Verbs: ``configure`` (line / split / body) · ``place(index)`` (reset episode
``index``, returns the task facts) · ``observe`` · ``goal_spec`` · ``step`` ·
``step_pose`` · ``panorama`` · ``submit_answer`` · ``judge_prompt`` ·
``evaluate`` · ``info``. A manipulation line is served by its own manager
behind the same object — ``_MANIP_ROUTES`` maps the line's prefix to it
(``libero-*`` -> ``manip.LiberoManager``, one Franka Panda — the LIBERO suites,
LIBERO-PRO and LIBERO-Plus alike, one engine and one verb surface, 2026-09-10;
``robotwin-*`` -> ``manip.RobotwinManager``, RoboTwin's two arms, 2026-09-10;
``robocasa-*`` / ``robocasa365-*`` -> ``manip_mobile.RobocasaManager``, a Panda
on an Omron base, 2026-09-10; ``calvin-*`` -> ``manip_chain.CalvinManager``, a
Panda whose episode is a CHAIN of five instructions revealed one at a time,
2026-09-10; ``behavior-*`` -> ``manip_mobile.BehaviorManager``, the BEHAVIOR-1K
challenge's R1 Pro — two arms on a holonomic base, 2026-09-10) — with ``place`` /
``observe`` / ``evaluate`` as
above plus the arm verbs ``state`` · ``move_ee`` · ``gripper`` · ``stop`` (the
so101 bench's, ``arm`` on every one) and ``move_base`` where the robot really
drives. The navigation verbs refuse on a manipulation line and the arm verbs
refuse on a navigation one.

``EmbodiedScoreEnvManager`` below is the bareES nodeset's manager (2026-09-08,
itself env_embodiedscore's) carried over verbatim — engine awareness (the
VLNverse lines on Isaac), GOAT image sub-goals rendered into ``goal_spec``,
EQA scoring folded into ``evaluate``, the EXPRESS judge prompt from
``prompts/evaluation.txt``, ``step_pose`` refused on the Isaac lines. Task
facts are the dataset's, verbatim. Episode addressing is file order, except
the HM3D lines whose boards were indexed through habitat-lab's seed-100
shuffle (objectnav-*, ovon, goat). Body overrides reach ``es.make(body=...)``
as ``dataclasses.replace`` on the variant's preset.
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import io
import json
import logging
import os
import random
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from core.envserver import ndarray_wire, png_b64

log = logging.getLogger("bareES.mcp")

HABITAT_SHUFFLE_SEED = (
    100  # habitat-lab EpisodeIterator(seed=config.seed) — the order the HM3D boards indexed
)
# Lines whose standing boards were indexed through habitat-lab's shuffled iterator.
_SHUFFLED_LINES = frozenset(
    {"objectnav-hm3d-v1", "objectnav-hm3d-v2", "objectnav-mp3d-v1", "ovon", "goat"}
)
_VARIANTS = ("standard", "upstream")
_DEFAULT_LINE = os.environ.get("EMBODIEDSCORE_LINE", "vlnce-r2r")
_DEFAULT_VARIANT = os.environ.get("EMBODIEDSCORE_VARIANT", "standard")
_DEFAULT_SPLIT = os.environ.get(
    "EMBODIEDSCORE_SPLIT", ""
)  # blank = the line's first declared split
_DEFAULT_GPU_ID = int(os.environ.get("EMBODIEDSCORE_GPU_ID", "0"))
_ACTION_STOP = 0
_ACTION_NAMES = ("STOP", "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN", "SUBTASK_STOP")
# EXPRESS-Bench's gpt-4o-mini judge prompt, vendored verbatim (env_express/prompts/evaluation.txt).
_JUDGE_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "express_judge_prompt.txt"
)
# The manipulation route table: line prefix -> (module of this package, manager class) that serves it.
# A new manipulation engine is one row here plus its manager.
_MANIP_ROUTES = {
    "libero-": ("manip", "LiberoManager"),  # LIBERO, LIBERO-PRO, LIBERO-Plus: a fixed-base Panda
    "robotwin-": ("manip", "RobotwinManager"),  # RoboTwin 2.0: two arms
    "robocasa-": ("manip_mobile", "RobocasaManager"),  # RoboCasa v0.2: a Panda on an Omron base
    "robocasa365-": ("manip_mobile", "RobocasaManager"),  # RoboCasa365: the same robot
    "calvin-": ("manip_chain", "CalvinManager"),  # CALVIN: a Panda on a chain of five instructions
    "behavior-": (
        "manip_mobile",
        "BehaviorManager",
    ),  # BEHAVIOR-1K: an R1 Pro, two arms on a holonomic base
}


def _repo_root() -> str:
    # __file__ lives at exp_workspace/bareES/mcp/env.py — three parents.
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )


def _data_roots() -> tuple[str, str]:
    """EmbodiedScore-envs data roots: the env vars, else <repo>/data/embodiedscore/{datasets,scenes} (a symlink farm, gitignored)."""
    root = _repo_root()
    return (
        os.environ.get("EMBODIEDSCORE_DATA_ROOT")
        or os.path.join(root, "data", "embodiedscore", "datasets"),
        os.environ.get("EMBODIEDSCORE_SCENE_ROOT")
        or os.path.join(root, "data", "embodiedscore", "scenes"),
    )


def _legacy_scene_id(scene_file: str) -> str:
    """The scene path as habitat-lab reported it (``data/scene_datasets/...``)."""
    _, scene_root = _data_roots()
    scene_root = os.path.abspath(scene_root)
    f = os.path.abspath(scene_file)
    rel = os.path.relpath(f, scene_root) if f.startswith(scene_root) else os.path.basename(f)
    return os.path.join("data", "scene_datasets", rel)


def habitat_shuffle_order(n: int, seed: int = HABITAT_SHUFFLE_SEED) -> list[int]:
    """File-order indices in the order habitat-lab's EpisodeIterator left the
    dataset list after its in-place ``random.shuffle`` (seed = habitat.seed)."""
    order = list(range(n))
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(order)
    return order


def lines() -> list[str]:
    """The benchmark lines the package declares (each in both variants)."""
    from embodiedscore_envs.benchmarks import BENCHMARKS

    return sorted({b.line for b in BENCHMARKS.values()})


def _disk_splits(line: str, data_root: str) -> list[str]:
    """Splits present on disk beyond the declaration — the workspace's derived
    files (rand100 / heldout100 / mip100 ...), which the package's loaders read
    by name."""
    root = Path(data_root)
    found: list[str] = []
    if line in ("vlnce-r2r", "vlnce-rxr"):
        base = (
            root
            / "vlnce"
            / ("R2R_VLNCE_v1-3_preprocessed" if line == "vlnce-r2r" else "RxR_VLNCE_v0")
        )
        suffix = ".json.gz" if line == "vlnce-r2r" else "_guide.json.gz"
        if base.is_dir():
            found = sorted(
                d.name for d in base.iterdir() if d.is_dir() and (d / f"{d.name}{suffix}").is_file()
            )
    elif line.startswith("objectnav-"):
        base = root / "objectnav" / line[len("objectnav-") :].replace("-", "/")
        if base.is_dir():
            found = sorted(
                d.name for d in base.iterdir() if d.is_dir() and (d / f"{d.name}.json.gz").is_file()
            )
    elif line in ("hmeqa", "mthm3d"):
        base = root / ("hmeqa" if line == "hmeqa" else "mt_hm3d")
        found = sorted(
            p.name[len("questions_") : -len(".csv")] for p in base.glob("questions_*.csv")
        )
    elif line == "express":
        found = sorted(
            p.name[len("express-bench_") : -len(".json")]
            for p in (root / "express_bench").glob("express-bench_*.json")
        )
    return found


def _default_split(splits: list[str]) -> str:
    """The evaluation split a fresh panel opens on: the unseen validation
    split where the line has one, else its validation split, else the first
    declared (a bare ``train`` default would load ~11k R2R episodes at boot)."""
    for name in ("val_unseen", "val", "val_seen"):
        if name in splits:
            return name
    return splits[0]


def _finite_or_none(value: Any) -> float | None:
    # inf (unreachable goal) is not JSON-serializable, so it leaves the wire as None.
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _json_safe(value: Any) -> Any:
    """Metrics / info values as they may leave the wire: non-finite floats
    become None, numpy scalars and arrays become Python values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    return value


def _goal_dict(goal: Any) -> dict[str, Any]:
    """The goal as JSON-safe data: its kind and fields, object instances trimmed
    to a count (an ObjectNav goal can carry hundreds of view points)."""
    if goal is None:
        return {"kind": "none"}
    kind = getattr(goal, "kind", "")
    if kind == "point":
        return {"kind": kind, "position": list(goal.position), "radius": float(goal.radius)}
    if kind == "object":
        return {"kind": kind, "category": goal.category, "n_instances": len(goal.instances)}
    if kind == "image":
        return {
            "kind": kind,
            "category": goal.instance.category,
            "camera_position": list(goal.camera_position),
            "camera_rotation": list(goal.camera_rotation),
            "hfov_deg": float(goal.hfov_deg),
            "image_size": list(goal.image_size),
        }
    if kind == "description":
        return {"kind": kind, "category": goal.instance.category, "description": goal.description}
    if kind == "question":
        return {
            "kind": kind,
            "text": goal.text,
            "answer": goal.answer,
            "choices": list(goal.choices) if goal.choices else None,
            "target": None if goal.target is None else _goal_dict(goal.target),
        }
    if kind == "sequence":
        return {"kind": kind, "n_goals": len(goal.goals), "kinds": [g.kind for g in goal.goals]}
    return {"kind": kind}


def _task_text(episode: Any, goal: Any) -> str:
    """The task statement the dataset gives — never a prompt of our own."""
    if episode.instruction:
        return str(episode.instruction)
    kind = getattr(goal, "kind", "")
    if kind == "question":
        return str(episode.info.get("question_formatted") or goal.text)
    if kind == "object":
        return str(goal.category)
    if kind == "description":
        return str(goal.description)
    if kind == "image":
        return str(goal.instance.category)
    return ""


def _load_judge_prompt() -> tuple[str, str]:
    """Mirror of EXPRESS upstream ``gpt.py:prompt_make`` on the vendored prompt
    (env_express): system = line index 1, user = lines 3..end; the caller
    appends the per-episode "Question/Answer/Response/Your mark:" block."""
    with open(_JUDGE_PROMPT_PATH, encoding="utf-8") as f:
        txt = f.readlines()
    prompt_system = txt[1]
    prompt = txt[3]
    for i in range(4, len(txt)):
        prompt = prompt + txt[i]
    return prompt_system, prompt


def _parse_judge_marks(judge_text: str) -> tuple[float, int] | None:
    """Parse the judge's "δ, σ" reply — mirror of EXPRESS evaluation.py:8-9
    (env_express). (grounding δ, correctness σ) or None when unparseable;
    upstream-exact split first, numeric-extraction fallback for chatty judges."""
    raw = str(judge_text or "").replace("Your mark:", "").strip()
    try:
        parts = raw.split(",")
        return float(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if len(nums) >= 2:
        try:
            return float(nums[0]), int(float(nums[1]))
        except ValueError:
            return None
    return None


def _apply_body_overrides(body: Any, overrides: dict[str, Any]) -> Any:
    """``dataclasses.replace`` on a Body preset; ``rgb`` / ``depth`` / ``navmesh``
    accept dicts of their own fields (partial: the preset's camera is the
    base when it has one), ``depth: null`` removes the depth camera."""
    from embodiedscore_envs.benchmarks.env import CameraSpec, NavMesh

    kw: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in ("rgb", "depth") and isinstance(value, dict):
            spec = dict(value)
            if "position" in spec:
                spec["position"] = tuple(float(x) for x in spec["position"])
            base = getattr(body, key, None)
            kw[key] = dataclasses.replace(base, **spec) if base is not None else CameraSpec(**spec)
        elif key == "navmesh" and isinstance(value, dict):
            kw[key] = NavMesh(**value)
        else:
            kw[key] = value
    return dataclasses.replace(body, **kw)


def _parse_resolution(value: Any) -> tuple[int, int] | None:
    """``task.rgb_resolution`` -> (height, width): 512 | "512" | "480x640" |
    [480, 640]; None / "" = the preset's own."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    s = str(value).lower()
    if "x" in s:
        h, w = s.split("x", 1)
        return int(h), int(w)
    n = int(s)
    return n, n


def _merge_body(
    block: dict[str, Any] | None, call: dict[str, Any] | None, rgb_resolution: Any = None
) -> dict[str, Any]:
    """The run's Body overrides: the env block's ``body`` under the call's
    ``body``, then ``rgb_resolution`` onto ``rgb`` (height / width only;
    hfov / position stay whatever the layers below say)."""
    out: dict[str, Any] = {**(block or {}), **(call or {})}
    hw = _parse_resolution(rgb_resolution)
    if hw is not None:
        rgb = dict(out.get("rgb") or {})
        rgb["height"], rgb["width"] = hw
        out["rgb"] = rgb
    return out


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """What the stack should be built for. The manager rebuilds only when the
    live spec differs (one rebuild for a line+split retarget)."""

    line: str
    variant: str
    split: str
    body_json: str = ""  # JSON Body overrides, "" = the variant's preset


# ══════════════════════════════════════════════════════════════════════
# EmbodiedScoreEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class EmbodiedScoreEnvManager:
    """One EmbodiedScore-envs stack at a time, for whichever (line, variant,
    split) the panel asks for.

    All public methods are blocking — call via
    ``asyncio.get_running_loop().run_in_executor(mgr.executor, fn)``.
    Single-thread executor enforces GL thread affinity.

    Two-stage laziness: ``_ensure_built`` loads the episodes (``es.make`` —
    no simulator yet), ``_ensure_placed`` creates the simulator on the first
    placement. A panel that retargets line and then split therefore costs two
    loader calls and one simulator build.
    """

    _instance: EmbodiedScoreEnvManager | None = None

    def __init__(self) -> None:
        self._env: Any = None
        self._benchmark: Any = None
        self._order: list[int] = []
        self._current_obs: dict | None = None
        self._info: dict = {}
        self._placed: bool = False
        self._step_budget: int | None = None  # the per-scene budget the reset info reported
        self._episode_done: bool = False
        self._step_count: int = 0
        self._last_action: int | None = None
        self._answer: str | None = None
        self._goal_image_cache: dict[tuple, str] = {}  # (episode index, goal index) -> base64 PNG
        self._lock = threading.RLock()
        self._spec: EnvSpec | None = None  # desired
        self._live: EnvSpec | None = None  # what _env was built for
        self._episode_index: int = 0
        self._gpu_id: int = _DEFAULT_GPU_ID
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="embodiedscore",
        )

    @classmethod
    def get(cls) -> EmbodiedScoreEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._spec is not None

    @property
    def spec(self) -> EnvSpec | None:
        return self._spec

    @property
    def _base(self) -> Any:
        return self._env.unwrapped

    @property
    def _world(self) -> Any:
        return self._env.unwrapped.world

    @property
    def _isaac(self) -> bool:
        return bool(self._benchmark is not None and self._benchmark.engine == "isaac")

    @property
    def episodes(self) -> list:
        eps = self._env.unwrapped.episodes
        return [eps[i] for i in self._order]

    # ── Lifecycle ──

    def initialize(
        self,
        line: str = _DEFAULT_LINE,
        variant: str = _DEFAULT_VARIANT,
        split: str = _DEFAULT_SPLIT,
        gpu_id: int = _DEFAULT_GPU_ID,
        body: dict[str, Any] | None = None,
    ) -> dict:
        """Record the spec and load its episodes (the simulator opens on the
        first placement)."""
        with self._lock:
            if self._spec is not None:
                log.warning("EmbodiedScoreEnvManager already initialized — skipping")
                return self.get_episode_info()
            self._gpu_id = int(gpu_id)
            self.configure(
                line=line,
                variant=variant,
                split=split or None,
                body_json=json.dumps(body, sort_keys=True) if body else "",
            )
            self._ensure_built_unlocked()
            return self.get_episode_info()

    def configure(
        self,
        line: str | None = None,
        variant: str | None = None,
        split: str | None = None,
        body_json: str | None = None,
    ) -> EnvSpec:
        """Update the desired spec (no build). A line change with no split
        named takes the new line's first declared split."""
        with self._lock:
            cur = self._spec
            new_line = line or (cur.line if cur else _DEFAULT_LINE)
            new_variant = variant or (cur.variant if cur else _DEFAULT_VARIANT)
            if new_variant not in _VARIANTS:
                raise ValueError(f"variant must be one of {_VARIANTS}, got {new_variant!r}")
            if split is None:
                split = (
                    cur.split
                    if (cur and cur.line == new_line)
                    else _default_split(self.list_splits(new_line))
                )
            new_body = body_json if body_json is not None else (cur.body_json if cur else "")
            self._spec = EnvSpec(new_line, new_variant, split, new_body)
            if self._live != self._spec:
                self._episode_index = 0
            return self._spec

    def _ensure_built_unlocked(self) -> None:
        if self._spec is None:
            raise RuntimeError("Environment not initialized")
        if self._env is not None and self._live == self._spec:
            return
        import embodiedscore_envs as es

        spec = self._spec
        if self._env is not None:
            self._env.close()  # no-op on a stack whose simulator never opened
            self._env = None
        b = es.benchmark(spec.line, spec.variant)
        if spec.split not in self.list_splits(spec.line):
            raise ValueError(
                f"{spec.line}: unknown split {spec.split!r} (known: {self.list_splits(spec.line)})"
            )
        data_root, scene_root = _data_roots()
        # The Isaac lines spawn a render worker that may run in a container
        # (scripts/isaac_container.sh): it mounts the roots the EMBODIEDSCORE_*
        # variables name and nothing else, so the roots THIS process resolved
        # (the workspace's symlink farm by default) must be visible to it —
        # else the stage "loads" empty (black frames, depth 20 m; 2026-09-08).
        os.environ.setdefault("EMBODIEDSCORE_DATA_ROOT", str(data_root))
        os.environ.setdefault("EMBODIEDSCORE_SCENE_ROOT", str(scene_root))
        kwargs: dict[str, Any] = {}
        if spec.body_json:
            kwargs["body"] = _apply_body_overrides(b.body, json.loads(spec.body_json))
        log.info(
            "Building EmbodiedScore-envs stack %s (%s) split=%s gpu=%d",
            b.name,
            spec.variant,
            spec.split,
            self._gpu_id,
        )
        self._env = es.make(
            spec.line,
            spec.split,
            variant=spec.variant,
            data_root=data_root,
            scene_root=scene_root,
            gpu_id=self._gpu_id,
            **kwargs,
        )
        self._benchmark = b
        n = len(self._env.unwrapped.episodes)
        self._order = habitat_shuffle_order(n) if b.line in _SHUFFLED_LINES else list(range(n))
        self._live = spec
        self._placed = False
        self._episode_done = False
        self._current_obs, self._info = None, {}
        self._goal_image_cache = {}
        log.info("Stack ready — %d episodes", n)

    def _ensure_placed_unlocked(self) -> None:
        self._ensure_built_unlocked()
        if not self._placed:
            self._place_unlocked(self._episode_index)

    def shutdown(self) -> None:
        with self._lock:
            if self._env is not None:
                log.info("Shutting down EmbodiedScore-envs stack")
                self._env.close()
                self._env = None
            self._live = None
            self._placed = False

    # ── Catalogue (env panel) ──

    def list_lines(self) -> list[str]:
        return lines()

    def list_variants(self) -> list[str]:
        return list(_VARIANTS)

    def list_splits(self, line: str | None = None) -> list[str]:
        import embodiedscore_envs as es

        line = line or (self._spec.line if self._spec else _DEFAULT_LINE)
        declared = list(es.benchmark(line).splits)
        data_root, _ = _data_roots()
        return declared + [s for s in _disk_splits(line, data_root) if s not in declared]

    def describe(self, line: str) -> str:
        import embodiedscore_envs as es

        return es.benchmark(line).description

    def action_table(self) -> list[str]:
        """The live action table by name (empty for a pose protocol)."""
        with self._lock:
            self._ensure_built_unlocked()
            if self._benchmark.pose or self._benchmark.polar:
                return []
            return [_ACTION_NAMES[int(a)] for a in self._base.actions]

    def body_summary(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_built_unlocked()
            b = self._base.body
            if (
                self._isaac
            ):  # IsaacBody: a camera on an occupancy grid — no tilt, no cylinder, no navmesh
                return {
                    "forward_step_m": b.forward_step_m,
                    "turn_angle_deg": b.turn_deg,
                    "tilt_angle_deg": None,
                    "tilt_limit_deg": None,
                    "locomotion": "kinematic",
                    "allow_sliding": False,
                    "agent_height_m": b.camera_height_m,
                    "agent_radius_m": None,
                    "rgb": dataclasses.asdict(b.rgb),
                    "depth": None if b.depth is None else dataclasses.asdict(b.depth),
                    "navmesh": f"occupancy ({b.collision})",
                    "pose_protocol": bool(self._benchmark.polar),
                    "engine": "isaac",
                }
            return {
                "forward_step_m": b.forward_step_m,
                "turn_angle_deg": b.turn_deg,
                "tilt_angle_deg": b.tilt_deg,
                "tilt_limit_deg": b.tilt_limit_deg,
                "locomotion": b.locomotion,
                "allow_sliding": b.allow_sliding,
                "agent_height_m": b.agent_height_m,
                "agent_radius_m": b.agent_radius_m,
                "rgb": dataclasses.asdict(b.rgb),
                "depth": None if b.depth is None else dataclasses.asdict(b.depth),
                "navmesh": b.navmesh.kind,
                "pose_protocol": bool(self._benchmark.pose),
                "engine": "habitat",
            }

    def get_total_episodes(self) -> int:
        with self._lock:
            if self._spec is None:
                return 0
            self._ensure_built_unlocked()
            return len(self._order)

    def get_episodes_list(self, offset: int = 0, limit: int = 50) -> dict:
        with self._lock:
            if self._spec is None:
                return {"episodes": [], "total": 0}
            self._ensure_built_unlocked()
            eps = self.episodes
            out = []
            for i, ep in enumerate(eps[offset : offset + limit]):
                goal = ep.goal
                if goal.kind == "sequence":
                    task = f"{len(goal.goals)} subtasks"
                else:
                    task = _task_text(ep, goal)
                out.append(
                    {
                        "index": offset + i,
                        "episode_id": str(ep.episode_id),
                        "scene_id": _legacy_scene_id(ep.scene.scene_file),
                        "goal_kind": goal.kind,
                        "task": task,
                    }
                )
            return {"episodes": out, "total": len(eps)}

    # ── Episode control (env panel + reset) ──

    def _place_unlocked(self, index: int) -> None:
        obs, info = self._env.reset(options={"episode": self._order[index]})
        self._current_obs, self._info = obs, info
        b = info.get("step_budget")  # only the reset info carries it
        self._step_budget = int(b) if b is not None else None
        self._placed = True
        self._episode_done = False
        self._step_count = 0
        self._last_action = None
        self._answer = None
        self._episode_index = index

    def set_episode_by_index(self, index: int) -> dict:
        """Place + arm episode ``index``; resets counters."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_built_unlocked()
            if index < 0 or index >= len(self._order):
                return {"error": f"Index {index} out of range (0-{len(self._order) - 1})"}
            self._place_unlocked(index)
            return self._get_episode_info_unlocked()

    def ensure_live(self) -> dict:
        """Template §5.1 reset semantics: a live episode is read untouched;
        a done one is re-armed at the SAME placement. Never chooses."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if not self._episode_done:
                return self._get_episode_info_unlocked()
            return self.set_episode_by_index(max(self._episode_index, 0))

    # ── Transition ──

    def _step_unlocked(self, action: Any) -> dict:
        obs, _r, terminated, truncated, info = self._env.step(action)
        self._current_obs, self._info = obs, info
        self._step_count += 1
        self._last_action = int(action) if np.ndim(action) == 0 else None
        self._episode_done = bool(terminated or truncated)
        return {"terminated": bool(terminated), "truncated": bool(truncated)}

    def _step_info_unlocked(self, **extra: Any) -> dict:
        info: dict[str, Any] = {
            "step_count": self._step_count,
            "collided": bool(self._info.get("collided", False)),
            "distance_to_goal": _finite_or_none(self._info.get("distance_to_goal")),
            "goal_index": int(self._info.get("goal_index", 0)),
            "subtask_closed": bool(self._info.get("subtask_closed", False)),
            "step_budget": self._step_budget_unlocked(),
        }
        info.update(self._get_agent_state_unlocked())
        info.update(extra)
        if self._episode_done:
            info["metrics"] = self._flat_metrics_unlocked()
        return info

    def step(self, action: int) -> dict:
        """Advance one tick with a discrete action. Control signals only."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if self._episode_done:
                return {"error": "Episode already done", "terminated": True}
            if self._benchmark.pose or self._benchmark.polar:
                return {"error": f"{self._benchmark.name} is a pose protocol — use step_pose"}
            action = int(action)
            if action >= len(self._base.actions):
                return {"error": f"action {action} not in this line's table {self.action_table()}"}
            flags = self._step_unlocked(action)
            return {"reward": 0.0, **flags, "info": self._step_info_unlocked(action=action)}

    def step_pose(
        self,
        target: list,
        yaw: float | None = None,
        goal_radius: float = 0.36,
        max_nav_steps: int = 50,
    ) -> dict:
        """Move toward a world-frame target. Pose protocols teleport there in
        one env step (``[x, z, yaw]``; yaw defaults to the current heading).
        Discrete lines walk with the greedy geodesic follower — real
        primitives, each charged to the budget and the metrics — stopping at
        ``goal_radius``, ``max_nav_steps``, episode end or path exhaustion;
        STOP / SUBTASK_STOP are never dispatched."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            if self._episode_done:
                return {"error": "Episode already done", "terminated": True}
            if self._isaac:
                return {
                    "error": f"{self._benchmark.name} runs on Isaac — no navmesh follower; use step"
                }
            goal = np.asarray(target, dtype=np.float32).reshape(3)
            if self._benchmark.pose:
                heading = float(self._world.heading()) if yaw is None else float(yaw)
                flags = self._step_unlocked(np.array([goal[0], goal[2], heading], dtype=np.float64))
                extra = {
                    "nav_steps": 1,
                    "step_geodesic": _finite_or_none(self._info.get("step_geodesic")),
                    "snapped": bool(self._info.get("snapped", False)),
                }
                return {"reward": 0.0, **flags, "info": self._step_info_unlocked(**extra)}

            follower = self._world.follower(float(goal_radius), fresh=True)
            steps = 0
            flags = {"terminated": False, "truncated": False}
            while steps < int(max_nav_steps) and not self._episode_done:
                try:
                    action = follower.next_action(goal)
                except Exception as e:  # FollowerError and friends: the walk just ends
                    log.warning("step_pose: follower failed: %s", e)
                    break
                if action is None:
                    break
                flags = self._step_unlocked(int(action))
                steps += 1
                pos = np.asarray(self._info["position"], dtype=np.float32)
                if float(np.linalg.norm(pos - goal)) < float(goal_radius):
                    break
            pos = np.asarray(self._info["position"], dtype=np.float32)
            extra = {"nav_steps": steps, "euclidean_to_target": float(np.linalg.norm(pos - goal))}
            return {"reward": 0.0, **flags, "info": self._step_info_unlocked(**extra)}

    def submit_answer(self, answer: str) -> dict:
        """EQA lines: record the answer and end the episode (STOP on a
        discrete line; a pose protocol never terminates, so it is marked done
        here). The comparison is the harness's — ``evaluate`` reports both."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            goal = self._base.episode.goal
            if goal.kind != "question":
                return {
                    "error": f"{self._benchmark.name} has no question to answer (goal kind {goal.kind!r})"
                }
            self._answer = str(answer)
            flags = {"terminated": self._episode_done, "truncated": False}
            if not self._episode_done:
                if self._benchmark.pose:
                    self._episode_done = True
                    flags["terminated"] = True
                else:
                    flags = self._step_unlocked(_ACTION_STOP)
            return {
                "answer": self._answer,
                "answer_gt": str(goal.answer),
                **flags,
                "info": self._step_info_unlocked(answer=self._answer),
            }

    # ── Perception (pull) ──

    def observe(self) -> dict:
        """Idempotent read of the current frame — never advances the env."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            obs = self._current_obs or {}
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            return {
                "rgb": np.asarray(rgb, dtype=np.uint8) if rgb is not None else None,
                "depth": np.asarray(depth, dtype=np.float32).squeeze()
                if depth is not None
                else None,
                "pose": self._get_agent_state_unlocked(),
                "intrinsics": self._cam_intrinsics_unlocked(),
                "depth_units": self._depth_units_unlocked(),
                "heading": float(self._world.heading()),
                "pitch": float(self._world.pitch()),
            }

    def goal_spec(self) -> dict:
        """The goal in force now (a GOAT sequence's cursor, else the episode goal)."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            return self._goal_spec_unlocked()

    def _goal_spec_unlocked(self) -> dict:
        ep = self._base.episode
        goal = self._base.current_goal()
        seq = ep.goal if ep.goal.kind == "sequence" else None
        spec = {
            "task": _task_text(ep, goal),
            "goal": _goal_dict(goal),
            "goal_kind": getattr(goal, "kind", "none"),
            "goal_index": int(self._base.goal_index),
            "n_goals": len(seq.goals) if seq is not None else 1,
            "episode_over": bool(self._episode_done),
        }
        if getattr(goal, "kind", "") == "image":
            spec["goal_image"] = self._goal_image_unlocked(goal)
        return spec

    def _goal_image_unlocked(self, goal: Any) -> str:
        """GOAT image sub-goal: the target instance drawn from the dataset's
        stored camera pose (position / rotation / hfov / size) — goat-bench
        renders it at the sub-task start, the dataset ships no JPEG. Same
        call as ``env_goat``; cached per (episode, sub-goal)."""
        key = (int(self._base.episode.index), int(self._base.goal_index))
        cached = self._goal_image_cache.get(key)
        if cached is None:
            from embodiedscore_envs.benchmarks.env import CameraSpec

            cam = CameraSpec(goal.image_size[1], goal.image_size[0], goal.hfov_deg)
            rgb = self._world.render_at(goal.camera_position, goal.camera_rotation, cam)["rgb"]
            cached = self._encode_rgb_base64(np.asarray(rgb, dtype=np.uint8))
            self._goal_image_cache[key] = cached
        return cached

    def _views_unlocked(self, n_views: int) -> list[tuple[int, float, dict]]:
        if self._isaac:  # the worker renders n headings from the pose; view 0 is the front
            rendered = self._world.render_views(int(n_views))
            return [(i, round(i * 360.0 / n_views, 1), obs) for i, obs in enumerate(rendered)]
        pos, rot = self._world.pose()
        q = np.quaternion(rot[3], rot[0], rot[1], rot[2])
        angle_step = 2.0 * np.pi / n_views
        out = []
        for i in range(n_views):
            yaw = i * angle_step
            new_rot = q * np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
            obs = self._world.render_at(pos, [new_rot.x, new_rot.y, new_rot.z, new_rot.w])
            out.append((i, round(float(np.degrees(yaw)) % 360, 1), obs))
        return out

    def render_panorama_rgbd(self, n_views: int = 12) -> dict:
        """Aligned RGB+Depth views at n_views headings from the current pose (read-only)."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            views = []
            for i, heading, obs in self._views_unlocked(n_views):
                view: dict[str, Any] = {"dir_id": i, "heading_deg": heading}
                view["rgb_base64"] = self._encode_rgb_base64(obs["rgb"])
                if "depth" in obs:
                    depth_m = np.squeeze(obs["depth"])
                    view["depth_base64"] = self._encode_depth_base64(depth_m)
                    view["depth_raw_base64"] = self._encode_depth_raw_base64(depth_m)
                views.append(view)
            return {"views": views, "n_views": len(views)}

    def render_panorama_composite(self, n_views: int = 12) -> dict:
        """Stitched labelled grid of n_views RGB headings (single image)."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            images, meta = [], []
            for i, heading, obs in self._views_unlocked(n_views):
                images.append(np.asarray(obs["rgb"], dtype=np.uint8)[:, :, :3])
                meta.append({"dir_id": i, "heading_deg": heading, "direction": f"{int(heading)}°"})
            composite = self._build_composite(images, [m["direction"] for m in meta])
            return {"views": meta, "n_views": len(meta), "composite": composite}

    @staticmethod
    def _encode_rgb_base64(rgb: np.ndarray) -> str:
        from PIL import Image

        img = Image.fromarray(rgb[:, :, :3].astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _encode_depth_base64(depth: np.ndarray) -> str:
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

    @staticmethod
    def _encode_depth_raw_base64(depth_m: np.ndarray) -> str:
        # 16-bit PNG depth in millimetres — preserves absolute metric depth
        # (render_at returns metres regardless of the stack's DepthClip).
        from PIL import Image

        d = np.squeeze(depth_m).astype(np.float32)
        d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        buf = io.BytesIO()
        Image.fromarray(d_mm, mode="I;16").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _build_composite(images: list, labels: list) -> np.ndarray | None:
        from PIL import Image, ImageDraw

        n = len(images)
        if n == 0:
            return None
        h, w = images[0].shape[:2]
        label_h, border = 20, 2
        if n <= 4:
            cols, rows = 2, 2
        elif n <= 8:
            cols, rows = 4, 2
        elif n <= 12:
            cols, rows = 4, 3
        else:
            cols, rows = 6, 4
        cell_w, cell_h = w + border * 2, h + label_h + border * 2
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for i, (img_arr, label) in enumerate(zip(images, labels, strict=True)):
            row, col = divmod(i, cols)
            x, y = col * cell_w + border, row * cell_h + border
            draw.text((x + w // 2 - len(label) * 3, y + 2), label, fill=(255, 255, 100))
            canvas.paste(Image.fromarray(img_arr), (x, y + label_h))
        return np.asarray(canvas, dtype=np.uint8)

    def _get_agent_state_unlocked(self) -> dict:
        # the engine's own frame: habitat y up + (x, y, z, w); Isaac z up + (w, x, y, z)
        pos, rot = self._world.pose()
        return {
            "position": np.asarray(pos, dtype=float).tolist(),
            "orientation": [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],
        }

    def _cam_intrinsics_unlocked(self) -> dict | None:
        rgb = self._base.body.rgb
        w, h = int(rgb.width), int(rgb.height)
        f = (w / 2.0) / float(np.tan(np.radians(float(rgb.hfov_deg)) / 2.0))
        return {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}

    def _depth_units_unlocked(self) -> dict | None:
        """What the depth array means. The units ride WITH the frame: a
        consumer that has to ask elsewhere can be told something true a
        rebuild ago."""
        body = self._base.body
        if body.depth is None:
            return None
        spec = self._benchmark.depth
        cam_h = float(body.camera_height_m) if self._isaac else float(body.depth.position[1])
        if spec is None:  # raw metres (upstream rigs that keep depth unprocessed)
            return {
                "scale_m": 1.0,
                "min_depth_m": None,
                "max_depth_m": None,
                "normalized": False,
                "camera_height_m": cam_h,
            }
        return {
            "scale_m": float(spec.max_m - spec.min_m) if spec.normalize else 1.0,
            "min_depth_m": float(spec.min_m),
            "max_depth_m": float(spec.max_m),
            "normalized": bool(spec.normalize),
            "camera_height_m": cam_h,
        }

    # ── Metric sink ──

    def evaluate(self, judge_text: str = "") -> dict:
        """Pull current task metrics on demand — independent of done. On the
        free-answer EQA line (EXPRESS) ``judge_text`` is the gpt-4o-mini
        judge's "δ, σ" reply, folded into C / C* / E_path."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            return self._flat_metrics_unlocked(judge_text)

    def judge_prompt(self, pred_answer: str) -> dict:
        """EXPRESS: the benchmark's judge prompt (system + user) for the
        current episode — env_express's ``judge_prompt``, on the package's
        episode (question / reference answer)."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            goal = self._base.episode.goal
            if goal.kind != "question" or goal.choices:
                return {"error": f"{self._benchmark.name} has no free-form question to judge"}
            system, user = _load_judge_prompt()
            user = user + (
                f"Question: {goal.text}\n"
                f"Answer: {goal.answer}\n"
                f"Response: {str(pred_answer or '').strip()}\n"
                "Your mark: "
            )
            return {"system": system, "user": user}

    def _flat_metrics_unlocked(self, judge_text: str = "") -> dict:
        """The METRICS payload: the line's metric wrapper output (its own
        keys — VLN_KEYS / OBJECTNAV_KEYS / GOAT's composite set / the EQA
        path terms), JSON-safe, plus the step counter and, on a question
        goal, the submitted and reference answers and the answer score:
        multiple choice -> ``success`` = letter match (env_hmeqa's
        evaluate); free answer -> the EXPRESS judge terms when a reply is
        given (env_express's evaluate: C = 100·clip(δ·σ,0,5)/5,
        C* = 100·clip(σ,0,5)/5, E_path = C · l/max(p,l), judge_ok)."""
        m = _json_safe(dict(self._info.get("metrics") or {}))
        m["step_count"] = self._step_count
        ep = self._base.episode
        goal = ep.goal
        if goal.kind == "question":
            m["answer"] = self._answer
            m["answer_gt"] = str(goal.answer)
            if goal.choices:
                pred = str(self._answer or "").strip().upper()
                m["success"] = 1.0 if pred and pred == str(goal.answer).strip().upper() else 0.0
            elif judge_text:
                marks = _parse_judge_marks(judge_text)
                judge_ok = marks is not None
                delta, sigma = marks if judge_ok else (0.0, 0)
                c = 100.0 * min(max(delta * sigma, 0.0), 5.0) / 5.0
                c_star = 100.0 * min(max(float(sigma), 0.0), 5.0) / 5.0
                p_len = float(m.get("path_length") or 0.0)
                l_gt = float(ep.info.get("geodesic_distance") or 0.0)
                weight = (l_gt / max(p_len, l_gt)) if max(p_len, l_gt) > 0 else 1.0
                m.update(
                    {
                        "c": c,
                        "c_star": c_star,
                        "e_path": c * weight,
                        "delta": float(delta),
                        "sigma": float(sigma),
                        "judge_ok": 1.0 if judge_ok else 0.0,
                        "gt_geodesic": l_gt,
                        "d_t": m.get("distance_to_goal"),
                    }
                )
        return m

    # ── Queries ──

    def _step_budget_unlocked(self) -> int | None:
        static = self._benchmark.max_episode_steps
        if static is not None:
            return int(static)
        return self._step_budget

    def get_episode_info(self) -> dict:
        """Panel-facing episode bundle — JSON-safe."""
        with self._lock:
            if self._spec is None:
                return {"error": "Environment not initialized"}
            self._ensure_placed_unlocked()
            return self._get_episode_info_unlocked()

    def _get_episode_info_unlocked(self) -> dict:
        ep = self._base.episode
        spec = self._live
        info = {
            "episode_id": str(ep.episode_id),
            "scene_id": _legacy_scene_id(ep.scene.scene_file),
            "episode_index": self._episode_index,
            "line": spec.line,
            "variant": spec.variant,
            "split": spec.split,
            "benchmark": self._benchmark.name,
            "step_count": self._step_count,
            "done": self._episode_done,
            "step_budget": self._step_budget_unlocked(),
            "instruction": ep.instruction or "",
        }
        info.update(self._goal_spec_unlocked())
        if ep.goal.kind == "question":
            info["question"] = ep.goal.text
            info["choices"] = list(ep.goal.choices) if ep.goal.choices else None
        return info


# ══════════════════════════════════════════════════════════════════════
# The verb surface
# ══════════════════════════════════════════════════════════════════════


class EmbodiedScoreEnv:
    """The ``env`` object of an experiment: one manager, one verb per public
    method. Every method is blocking and must run on ONE thread (GL
    affinity) — core.envserver serves requests sequentially for that reason."""

    name = "EmbodiedScoreEnv"

    def __init__(
        self,
        variant: str = _DEFAULT_VARIANT,
        gpu_id: int = _DEFAULT_GPU_ID,
        body: dict[str, Any] | None = None,
        python: str | None = None,
        port: int | None = None,
        shapes: Any = None,
        env_fields: Any = None,
        robotwin_root: str | None = None,
        calvin_dataset: str | None = None,
        behavior_root: str | None = None,
    ) -> None:
        # python / port / shapes / env_fields are the launcher's and the
        # runner's keys of the same block; accepted so the block instantiates
        # as written. ``body``: Body overrides on the variant's preset
        # (forward_step_m, turn_deg, rgb: {width, height, hfov_deg}, depth: null …).
        self._variant = variant
        self._gpu_id = int(gpu_id)
        self._body: dict[str, Any] = dict(body or {})
        self._mgr = EmbodiedScoreEnvManager.get()
        self._loader: dict[str, Any] = {
            "robotwin_root": robotwin_root,
            "calvin_dataset": calvin_dataset,
            "behavior_root": behavior_root,
        }
        self._manip: Any = None  # the line's manipulation manager once one is configured

    @staticmethod
    def _manip_route(line: str) -> tuple[str, str] | None:
        """(module, class) of the manager serving ``line``, or None when it is a
        navigation line — the route table, one entry per manipulation engine."""
        for prefix, route in _MANIP_ROUTES.items():
            if str(line).startswith(prefix):
                return route
        return None

    @classmethod
    def _is_manip(cls, line: str) -> bool:
        return cls._manip_route(line) is not None

    def _nav_only(self, verb: str) -> dict | None:
        if self._manip is not None:
            return {
                "error": f"{verb} is a navigation verb; this run serves a manipulation line — use move_ee / gripper / stop"
            }
        return None

    def _manip_only(self, verb: str) -> dict | None:
        if self._manip is None:
            return {
                "error": f"{verb} is a manipulation verb; this run serves a navigation line — use step"
            }
        return None

    # ── placement of the whole run ──

    def configure(
        self,
        line: str,
        split: str,
        body: dict[str, Any] | None = None,
        rgb_resolution: Any = None,
        **_ignored: Any,
    ) -> dict:
        """Line + split (+ body overrides) for the run; the episodes load now,
        the simulator on the first ``place``. The body: the block's ``body``
        under the call's, then ``rgb_resolution`` (``task.rgb_resolution``
        forwarded by ``env_fields``: 512 | "480x640" -> body.rgb)."""
        if route := self._manip_route(line):
            import importlib
            import inspect

            module, cls_name = route
            if self._manip is None:
                module, cls_name = route
                manager = getattr(importlib.import_module("." + module, __package__), cls_name)
                # engine-specific loader kwargs reach only the manager that takes them (by name, or **kwargs)
                params = inspect.signature(manager).parameters
                takes_any = any(q.kind is inspect.Parameter.VAR_KEYWORD for q in params.values())
                loader = (
                    dict(self._loader)
                    if takes_any
                    else {k: v for k, v in self._loader.items() if k in params}
                )
                self._manip = manager(
                    variant=self._variant,
                    gpu_id=self._gpu_id,
                    body={**self._body, **(body or {})},
                    **loader,
                )
            return self._manip.configure(line=line, split=split, rgb_resolution=rgb_resolution)
        if self._manip is not None:
            return {
                "error": "this server already serves a manipulation line; one server, one engine"
            }
        m = self._mgr
        merged = _merge_body(self._body, body, rgb_resolution)
        body_json = json.dumps(merged, sort_keys=True) if merged else ""
        if not m.initialized:
            return m.initialize(
                line=line,
                variant=self._variant,
                split=split,
                gpu_id=self._gpu_id,
                body=merged or None,
            )
        m.configure(line=line, variant=self._variant, split=split, body_json=body_json)
        return m.get_episode_info()

    def info(self) -> dict:
        """Health / provenance: what is served."""
        if self._manip is not None:
            return self._manip.info()
        spec = self._mgr.spec
        return {
            "name": self.name,
            "line": spec.line if spec else None,
            "variant": spec.variant if spec else self._variant,
            "split": spec.split if spec else None,
            "episodes": self._mgr.get_total_episodes() if spec else 0,
        }

    # ── per episode ──

    def place(self, index: int) -> dict:
        """Reset episode ``index`` and return the task facts the briefing
        embeds (the former reset verb's payload)."""
        if self._manip is not None:
            return self._manip.place(int(index))
        meta = self._mgr.set_episode_by_index(int(index))
        if "error" in meta:
            return meta
        return {
            "task": str(meta.get("task", "")),
            "goal": meta.get("goal"),
            "episode_id": str(meta.get("episode_id", "")),
            "scene_id": str(meta.get("scene_id", "")),
            "line": str(meta.get("line", "")),
            "step_budget": meta.get("step_budget"),
            "goal_index": meta.get("goal_index", 0),
            "n_goals": meta.get("n_goals", 1),
        }

    def observe(self) -> dict:
        """The current frame on the wire: ``rgb`` as base64 PNG (what the
        bridges decode), ``depth`` as the lossless ``__ndarray__`` marker
        (float32 metres — PNG cannot hold it), the rest plain JSON. On a
        manipulation line: ``rgb`` (the agentview) + ``cameras`` {head, wrist}."""
        if self._manip is not None:
            return self._manip.observe()
        out = self._mgr.observe()
        if "error" in out:
            return out
        rgb, depth = out.pop("rgb"), out.pop("depth")
        out["rgb"] = png_b64(rgb) if rgb is not None else None
        out["depth"] = (
            ndarray_wire(np.asarray(depth, dtype=np.float32)) if depth is not None else None
        )
        return out

    def goal_spec(self) -> dict:
        return self._nav_only("goal_spec") or self._mgr.goal_spec()

    def step(self, action: int) -> dict:
        return self._nav_only("step") or self._mgr.step(int(action))

    def step_pose(
        self,
        target: list,
        yaw: float | None = None,
        goal_radius: float = 0.36,
        max_nav_steps: int = 50,
    ) -> dict:
        return self._nav_only("step_pose") or self._mgr.step_pose(
            [float(x) for x in target],
            None if yaw is None else float(yaw),
            float(goal_radius),
            int(max_nav_steps),
        )

    def panorama(self, n_views: int = 12, composite: bool = False) -> dict:
        if err := self._nav_only("panorama"):
            return err
        if composite:
            return self._mgr.render_panorama_composite(int(n_views))
        return self._mgr.render_panorama_rgbd(int(n_views))

    def submit_answer(self, answer: str) -> dict:
        return self._nav_only("submit_answer") or self._mgr.submit_answer(str(answer))

    def judge_prompt(self, pred_answer: str) -> dict:
        return self._nav_only("judge_prompt") or self._mgr.judge_prompt(str(pred_answer))

    # ── the manipulation lines' arm verbs (the line's manager; the so101 bench's signatures) ──

    def state(self) -> dict:
        return self._manip_only("state") or self._manip.state()

    def move_ee(
        self,
        arm: str,
        x: float,
        y: float,
        z: float,
        roll: float | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
        gripper: float | None = None,
    ) -> dict:
        return self._manip_only("move_ee") or self._manip.move_ee(
            arm, x, y, z, roll, pitch, yaw, gripper
        )

    def move_base(self, dx: float, dy: float, dyaw: float = 0.0) -> dict:
        if err := self._manip_only("move_base"):
            return err
        if not hasattr(self._manip, "move_base"):
            return {"error": "this line's robot has no mobile base — use move_ee"}
        return self._manip.move_base(dx, dy, dyaw)

    def gripper(self, arm: str, open: float) -> dict:
        return self._manip_only("gripper") or self._manip.gripper(arm, open)

    def stop(self) -> dict:
        return self._manip_only("stop") or self._manip.stop()

    def evaluate(self, judge_text: str = "", **_ignored: Any) -> dict:
        if self._manip is not None:
            return self._manip.evaluate()
        return self._mgr.evaluate(str(judge_text or ""))

    def close(self) -> None:
        if self._manip is not None:
            self._manip.close()
        self._mgr.shutdown()
