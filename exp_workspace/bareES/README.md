# bareES — the bare surface on EmbodiedScore-envs, every line

> 2026-09-09 — this folder is the experiment: `prompts.py`, the switches in its yaml, and `mcp/` — its
> MCP tool set: `env.py` (EmbodiedScoreEnv, the world as a verb object; the runner serves it with `core.envserver`
> under `ac-es` on :9200 and stops it when the run ends) + `bridge.py` (those verbs as the model's
> tools). The auto_host / nodeset / workers.sh instructions below predate that and are kept as
> history; run with `python runner.py <run name> [+run.fake=true] run.episodes=…`.

Folder = method arm; cells carry the arm word `bareES`
(`std_<task>_{root}_{model}_{tier}_bareES`). Built 2026-09-08 to run every
package line through bare's two-tool surface on the STANDARD body.

**What it is:** bare (`exp_workspace/bare`: observe + step over the stdio MCP
bridge, classic alternation, no instruments) with the environment half
replaced by EmbodiedScore-envs — episodes, bodies, task semantics, metrics
and budgets come from the package through this folder's `nodeset/` (the
frozen copy of `workspace/nodesets/env/env_embodiedscore`). One bridge, one
briefing builder and one driver branch serve all thirteen lines; the task
SHAPE is read off what reset() returns (`core.driver.es_task_of`), not off a
per-line table.

| | bare | bareES |
|---|---|---|
| env | `env_habitat` (habitat-lab 0.1.7, R2R / RxR) | `env_bare_es` ← EmbodiedScore-envs, 13 lines (habitat-sim 0.3.3 + Isaac Sim 5.1) |
| body | std R2R rig, actions 0–3 | the package's STANDARD variant everywhere: actions 0–5 (tilt 30°, ±60°); Isaac lines 0–3 (yaw-only worker); GOAT 0–6 |
| tools | observe · step | observe · step (+ `answer` on the EQA lines) |
| task shapes | nav | nav · objnav · goat · eqa · express |
| metrics | hand-written NE/SR/SPL/OSR/nDTW | the line's package keys; EQA letter match and the EXPRESS judge fold done in `nodeset/evaluate` |
| task / env keys | task r2r · env habitat | task r2r · env `es` (the run name reads `std_r2r_es_cc_…_bareES`; former line keys `es*` retired 2026-09-09) |

## Profiles (口径 inherited from the legacy arms; ★ = no precedent)

| profile | line / split / episodes | turns | shape |
|---|---|---|---|
| esr2r · esrxr | vlnce-r2r · vlnce-rxr / rand100 / 0-99 | 200 | nav |
| esivlnce ★ | ivlnce / val_unseen / 0-99 (tours = consecutive episodes; ONE worker, in order) | 200 | nav |
| eshm3d · esmp3d | objectnav-hm3d-v1 · objectnav-mp3d-v1 / mip100 / 0-99 | 150 | objnav |
| eshm3dv2 ★ | objectnav-hm3d-v2 / val / 0-99 (val_mini has 30) | 150 | objnav |
| esovonseen · syn · unseen | ovon / mip100_{seen,seen_synonyms,unseen} / 0-99 | 150 | objnav |
| esgoat ★ | goat / val_unseen / 0-99; 5000 steps | 150 × sub-goals (cap 1500); timeout 800 s × sub-goals (cap 8000) | goat |
| eshmeqa · esmthm3d | hmeqa · mthm3d / mip100 / 0-99 | 150 | eqa |
| esexpress | express / mip100 / 0-99 | 150 | express |
| esvlnverse · esvlnversecoarse ★ | vlnverse-fine · vlnverse-coarse / val_unseen / 0-99 | 200 | nav (4 actions) |

## GOAT on the bare surface

No new tool. SUBTASK_STOP is action 6 in `step()` (as STOP is 0). The
briefing gives the sub-goal count and the FIRST sub-goal only; every
later one is announced in the step result that closed its predecessor
— text for category / description sub-goals, the target's photo attached
as an image for image sub-goals (rendered from the dataset's stored camera
pose by `nodeset/goal_spec`, as goat-bench and `env_goat` do; the first
sub-goal's photo rides the first `observe()`). Closing the last sub-goal
ends the episode (`end_reason` = `subtasks_closed`).

## The manipulation lines (LIBERO and RoboTwin, 2026-09-10)

Both manipulation engines reach the agent through one verb layer. `mcp/env.py`
keeps a route table — line prefix -> manager (`_MANIP_ROUTES`: `libero-` ->
`LiberoManager`, `robotwin-` -> `RobotwinManager`) — and `mcp/manip.py`'s
`ManipManager` holds everything that does not depend on how many arms the robot
has. Every verb takes an `arm` (the so101 bench's signature), `state` reports
every arm, and `move_ee` / `gripper` name the one to act; a two-armed robot's
other arm holds its pose AND its last gripper command while one arm moves, so a
grasp is not dropped by moving the partner. The briefing (`prompts.py`, `manip`)
and the bridge's tool descriptions take the arm names from the placement
payload, through `EsManip.bridge_knobs` -> `BAREES_ARMS`.

### LIBERO

`std_libero_spatial_es_bareES.yaml` runs the package's `libero-spatial` line
(standard variant: `LiberoPoseEnv`, 256² agentview + wrist, 100 macro steps)
through the same `EmbodiedScoreEnv` object — `mcp/env.py` hands a `libero-*`
line to `mcp/manip.py` (`LiberoManager`) — behind the arm surface
`mcp/bridge_manip.py` (`surface.bridge` in the yaml): observe / state / move_ee /
gripper / stop, the so101 bench's tools, so the real arm and the simulator are
one interface. Shape `core.shapes.es.EsManip`; briefing `prompts.py` `manip`.
The env server runs under `ac-libero` (INSTALL-libero.md of the package) on
port 9201 — 9200 is the so101 tunnel's when the robot box is attached.

What the agent sees: two frames, the gripper pose (world frame, metres /
degrees) and opening, the budget, whether a move converged. What it never sees:
success (the package latches the first tick the BDDL goal held; the episode
ends only on stop() or the budget), object poses, predicates — `evaluate`
reports them driver-side. Verified 2026-09-10: a scripted rim grasp of the
bowl on `libero_spatial/0/0` reaches `success=1.0` in 8 macro steps;
`+run.fake=true` walks observe → stop through the whole runner.

### RoboTwin

`std_robotwin_clean_es_bareES.yaml` runs the package's `robotwin-clean` line
(standard variant: `RobotwinPoseEnv`, a 640×480 head camera and one wrist camera
per arm, 100 macro steps) through the same `EmbodiedScoreEnv` object and the
same `mcp/bridge_manip.py` surface. The robot is RoboTwin's `aloha-agilex`: two
6-DoF arms, `arm` = `left` | `right`. The env server runs under `ac-robotwin`
(INSTALL-robotwin.md of the package) on port 9310, and the yaml names the
RoboTwin checkout in `env.robotwin_root` so the experiment file stays the only
place a path is looked up.

What the agent sees: three frames (head + both wrists), each arm's gripper pose
and opening, the budget, whether a move converged. What it never sees: success
(the package latches the first action the task's own `check_success` held),
actor poses, the check's name — `evaluate` reports them driver-side. Verified
2026-09-10: RoboTwin's own scripted expert for `adjust_bottle`, driven through
the macro protocol by `scripts/robotwin_expert_episode.py` of the package,
reaches `success=1.0` in 7 macro steps (10 RoboTwin actions) on episode 0;
`+run.fake=true` on port 9310 walks observe → stop through the whole runner.

## The robustness lines (LIBERO-PRO · LIBERO-Plus, 2026-09-10)

Two 2025 benchmarks perturb LIBERO's tasks to ask whether a policy understood
them or memorised them. They are lines on the same engine, so they reach the
agent through exactly the same surface — same manager, same five tools, same
briefing, same shape. `mcp/env.py` routes them with the table `_MANIP_ROUTES`
(line prefix → manager); one row covers all three LIBERO benchmarks.

| yaml | line | split | episodes | env server |
|---|---|---|---|---|
| `std_libero_pro_spatial_es_bareES.yaml` | `libero-pro-spatial` | mini | 400 | `ac-libero-pro`, port 9300 |
| `std_libero_plus_camera_es_bareES.yaml` | `libero-plus-camera` | mini | 40 | `ac-libero-plus`, port 9301 |

Each benchmark ships its own `libero` package, so each has its own conda env
(INSTALL-libero.md § LIBERO-PRO, § LIBERO-Plus) and its own port; one server
serves one engine. LIBERO-PRO's line is a base suite carrying that suite's ten
tasks under four perturbations (object / position / semantic / task);
LIBERO-Plus's is one perturbation kind (camera viewpoint here) across all four
suites. Nothing else changes for the agent: the instruction, the two frames and
the arm verbs are what they are on `libero-spatial`.

Verified 2026-09-10: the scripted rim grasp reaches `success=1.0` in 9 macro
steps on the first episode of each (`libero_spatial_object/0/0` and
`libero_spatial/608/0`, through the package's
`scripts/libero_scripted_episode.py`); `+run.fake=true` walks observe → stop
through the runner on both.

## The RoboCasa lines (a mobile manipulator, 2026-09-10)

`std_robocasa365_atomic_es_bareES.yaml` runs the package's
`robocasa365-atomic-seen` line and `std_robocasa_pnp_es_bareES.yaml` the
RoboCasa v0.2 `robocasa-pnp` line (standard variant both:
`RobocasaPoseEnv`, 256² `robot0_agentview_center` + wrist, 100 macro steps),
through the same `EmbodiedScoreEnv` object. `mcp/env.py` routes by the table
`_MANIP_ROUTES` (line prefix -> manager module + class): `libero-*` to
`mcp/manip.py` (`LiberoManager`), `robotwin-*` to `mcp/manip.py`
(`RobotwinManager`), `robocasa-*` and `robocasa365-*` to `mcp/manip_mobile.py`
(`RobocasaManager`).

The robot is a Franka Panda on an Omron mobile base, so the surface is the
so101 verbs **plus the one this robot really has**:

    observe · state · move_ee · gripper · stop      the so101 bench's, unchanged
    move_base(dx, dy, dyaw)                          drive the base in its own frame
                                                     (metres / degrees on the wire)

`state()` reports the base pose next to the gripper's; `move_base` counts one
motion step like `move_ee`. The bridge is the same `mcp/bridge_manip.py`: the
`move_base` tool is registered only when `BAREES_BASE=1`, which the yaml sets
through `agent.arm.base: true` and `env_map: {BASE: base}`, so the LIBERO
surface is byte-for-byte what it was. The briefing is `prompts.py`'s `manip`
branch with its mobile-base paragraphs, switched on by the same fact:
`RobocasaManager.place()` returns `base: True` and `core.shapes.es.EsManip`
passes it to `build_briefing`.

The env server runs under `ac-robocasa365` (RoboCasa365) or `ac-robocasa`
(RoboCasa v0.2) on ports 9320 / 9321 — the two releases pin incompatible
numpy / mujoco / python and share a distribution name, so they cannot share an
interpreter (INSTALL-robocasa.md of the package). Every reset **generates a
kitchen**, which costs tens of seconds; the yamls' `episode_timeout`,
`server_timeout_s` and `client_timeout_s` are raised accordingly.

What the agent sees: two frames, the gripper pose and opening, the base pose,
the budget, whether a move converged. What it never sees: success (the package
latches the first tick RoboCasa's own `_check_success` held; the episode ends
only on stop() or the budget), object poses, fixtures — `evaluate` reports them
driver-side.

Verified 2026-09-10 on `robocasa365-atomic-seen` (`scripts/robocasa_scripted_episode.py`,
privileged information, no model): a scripted pick-and-place of the blender lid
on `CloseBlenderLid/1-1/0` — the line's first task — reaches `success=1.0` in 12
macro steps, and a scripted drive on `NavigateKitchen/1-1/0` reaches it in 4
`move_base` steps. `+run.fake=true` walks observe -> stop through the whole
runner on both files, port 9320 (RoboCasa365) and 9321 (v0.2). The RoboCasa v0.2
line is verified as far as the package's contracts and the runner go; no scripted
success was reached on it — the generic push policy aims at the fixture's origin,
not at the drawer face, so `CloseDrawer` stalls against the cabinet.

## The CALVIN line (a chain of five instructions, 2026-09-10)

`std_calvin_d_es_bareES.yaml` runs the package's `calvin-d` line (standard
variant: `CalvinPoseEnv`, 256² static + gripper cameras, 20 macro moves per
instruction) through the same `EmbodiedScoreEnv` object. `mcp/env.py` routes
`calvin-*` to `mcp/manip_chain.py` (`CalvinManager`) — the fourth row of
`_MANIP_ROUTES`.

The robot is the same shape as LIBERO's — one fixed-base Franka Panda called
"panda" — so the verb surface is unchanged: `observe · state · move_ee ·
gripper · stop`, no new tool. What is new is the **task**. A CALVIN episode is
five language instructions in a row:

    take the blue block and rotate it to the right
    push the sliding door to the right side
    lift the red block from the sliding cabinet
    store the grasped block in the sliding cabinet
    use the switch to turn off the light bulb

and the agent is told only the one in force. **The environment closes each
one, not the agent**: CALVIN's task oracle checks the scene every control tick
and the chain advances by itself, so there is no verb for declaring a
sub-task done. Instead the motion result that happened to finish one carries
the announcement, and `mcp/bridge_manip.py`'s `_announce` puts it at the front
of the tool result:

    {"subtask_closed": true,
     "NEW_INSTRUCTION": "push the sliding door to the right side",
     "message": "That completed the instruction. NEW INSTRUCTION (2 of 5): … .
                 Your move budget starts again for it.", …}

That is the GOAT lines' announcement through the tool-result channel
(`bridge.py`'s `_goal_content`) minus the declaration — on GOAT the agent asks
for the switch with SUBTASK_STOP and the bridge answers; here the oracle
decides and the answer rides whichever move did the job.

The **budget is per instruction**, not per episode: 20 macro moves each, and an
instruction that spends them without being solved ends the whole chain (CALVIN's
`evaluate_sequence` returns at the first failure). `move_ee` / `gripper`
therefore report `moves_remaining_on_this_instruction`, and the count restarts
when a new instruction arrives. `task.max_turns_per_goal` and
`task.episode_timeout_per_goal` scale the session caps with the chain, exactly
as they do on GOAT (`core.shapes.es.EsManip.caps_of`).

The briefing is `prompts.py`'s `manip` branch in its **chain** form, switched on
by the same fact the metrics use: `CalvinManager.place()` reports `n_goals = 5`,
and `core.shapes.es.es_goal` now carries a manip payload's goal count through
instead of flattening it to 1 (a manip goal with five sub-goals must still map
to `manip`, never to `goat`). The chain form replaces the `TASK:` line with the
sequence paragraph above, uses CALVIN's own table geometry (top at z = 0.46, the
sliding cabinet, drawer, button and switch at the back) and states the
per-instruction budget.

The env server runs under `ac-calvin` on port 9330 (calvin_env + pybullet
through pybullet's own EGL plugin; INSTALL-calvin.md of the package). Resets are
cheap — one scene for every episode, so a reset is a state write at ~0.01 s
after a ~1.2 s first build — but the **first server start generates CALVIN's
1000 evaluation sequences** (~1 min) and caches them under the data root, which
is why `run.server_timeout_s` is 900.

What the agent sees: two frames, the gripper pose and opening, the instruction
in force and its budget, whether a move converged. What it never sees: how many
instructions it has solved, the object poses, the initial condition — `evaluate`
reports them driver-side, together with the chain and `ChainMetrics`
(`chain_length`, `success_1..success_5`, whose means over a split are CALVIN's
average successful sequence length and its success rates for i instructions in
a row).

Verified 2026-09-10 on `calvin-d` (`scripts/calvin_scripted_episode.py`,
privileged information, no model): sequence 0 reaches **chain_length 2 of 5** in
19 macro steps — `rotate_blue_block_right` then `move_slider_right`, both closed
by CALVIN's own oracle — and the first ten sequences give chain lengths
2 · 0 · 1 · 2 · 2 · 0 · 2 · 1 · 1 (seven of nine with at least one instruction
solved). `+run.fake=true` walks observe -> stop through the whole runner on port
9330.

## The BEHAVIOR-1K line (two arms on a base, in a whole house, 2026-09-10)

`std_behavior_1k_es_bareES.yaml` runs the package's `behavior-1k` line — the
2025 BEHAVIOR Challenge, standard variant: `BehaviorPoseEnv`, 256² head + both
wrist cameras, 150 macro steps — through the same `EmbodiedScoreEnv` object.
`_MANIP_ROUTES` gains one row: `behavior-*` -> `mcp/manip_mobile.py`
(`BehaviorManager`). That module now holds the half both mobile managers share
(`MobileManipManager`: the stack's lifecycle, the episode guard, the budget
accounting, `move_base`, `stop`, `evaluate`) with `RobocasaManager` and
`BehaviorManager` filling in the arm-shaped verbs — one arm on RoboCasa, two on
BEHAVIOR. RoboCasa's behaviour is unchanged.

The robot is the challenge's R1 Pro: a holonomic wheeled base, a four-joint
torso, two 7-DOF arms with a parallel gripper each. So the surface is the so101
verbs, `arm` naming a real arm, plus the mobile base:

    observe · state · gripper · stop                 the so101 bench's, unchanged
    move_ee(arm, …)                                  arm is "left" or "right"; the other holds
    move_base(dx, dy, dyaw)                          drive the base in its own frame

`observe()` returns three frames (head, left wrist, right wrist); `state()`
reports both arms, whether each is holding something, the base pose and the
torso joints. No verb was added beyond RoboCasa's `move_base`.

The briefing is `prompts.py`'s `manip` branch in a new shape,
`_MANIP_TWO_ARMS_MOBILE` — the two-arm wording on a wheeled base, in a house
rather than at a table. It is chosen by the same two facts the other lines use:
`BehaviorManager.place()` reports `arms: ["left", "right"]` and `base: True`,
and `core.shapes.es.EsManip` passes both to `build_briefing`. The `move_base`
tool's sentence and the reach rule now say "both arms" when the robot has two.
The bridge (`mcp/bridge_manip.py`) needed no change at all: it already names
whatever arms `BAREES_ARMS` lists and registers `move_base` on `BAREES_BASE=1`.

The env server runs under `ac-behavior` on port 9340. It is the slowest stack
in the workspace: Isaac Sim 4.5 boots once per process (~30 s) and the first
`place()` of a scene model loads a whole furnished house, which took ~12
minutes here on a cold cache. The yaml's `server_timeout_s` (1800),
`client_timeout_s` (3600) and `episode_timeout` (7200) are sized for that. The
world reloads at the cheapest level a change allows — a new instance of the
same task replays a JSON state (instant); a new task, like a new house, clears
the stage and rebuilds (the challenge loads a scene file per task instance, so
`env.update_task` cannot serve a task change) — so episode order matters, and
the line is ordered task-major with the tasks grouped by house.

`env.behavior_root` names OmniGibson's data root (the encrypted asset bundle,
the robot assets and the challenge task instances, ~36 GB) and reaches both the
episode loader and the world; it is the one engine-specific yaml key, carried
the way `robotwin_root` is.

The macro protocol's own reach loop is the one place this engine differs
sharply from RoboCasa's. An `absolute_pose` IK command is read in the frame of
the controller's own reference pose, which on a holonomic base is the
ARTICULATION ROOT (`base_footprint_x`) — a stationary virtual link, not the
`base_footprint_link` that `robot.get_position_orientation()` returns. Anchoring
a world target on the wrong one makes every reach miss by exactly how far the
base has driven; the port composes through the controller's own reference
instead, so it never has to name the frame.

What the agent sees: three frames, both gripper poses and openings, the base
pose, the torso, the budget, whether a move converged. What it never sees:
success, the q_score, which goal predicates hold, the poses of the
task-relevant objects — `evaluate` reports all of them driver-side, alongside
the challenge's own efficiency terms.

Verified 2026-09-11 on `behavior-1k`, episode 0 = `turning_on_radio/242` in
`house_double_floor_lower`. The package's whole contract file passes with the
simulator tier switched on: `EMBODIEDSCORE_RUN_BEHAVIOR_TESTS=1 pytest
tests/test_behavior_contracts.py` -> **20 passed** in 33 minutes. Reset serves
256² head + both wrists + an 18-D proprio vector, the goal parses to BDDL's own
`radio_receiver1 is toggled_on`, and the tick cap is the challenge's 4299. Six
`move_base` moves drove the robot 2.9 m across the room to the radio, every one
converging inside 2-5 cm, and both grippers open and close on command.

The scripted press of the radio's toggle button still does not land: the button
is out of the arm's reach with the torso held fixed (see Deviations), the
closest approach was 0.247 m, and no non-zero q_score was reached. That is the
protocol's known gap, not a control fault — a reachable 5 cm move converges,
which is what the contract test checks.

`+run.fake=true` walks observe -> stop through the whole runner on port 9340,
registering all six tools and returning the challenge's metrics through
`evaluate` (`q_score`, per-part travel, `tick_cap` 4299, `n_predicates` 1). The
RoboCasa lines still pass their own fake run on the refactored manager.

## Deviations (documented, not hidden)

- STANDARD action table on every line (user decision 2026-09-08): the bare
  arm's R2R/RxR boards ran 0–3 — bareES nav cells are not pooled with them.
- HM-EQA / MT-HM3D / EXPRESS on the STANDARD body (level start, 15° turns),
  not the lines' own rigs; same as slam_02es.
- ObjectNav briefing says "right next to it (within about 1 m)"; the
  package's success rule is the line's (0.1 m to a view point / 0.25 m OVON).
- VLNverse: the zero-shot line's kinematic camera on the STANDARD numbers
  (`bodies.VLNVERSE_STANDARD`), depth 256² resampled from the 512² render.

- LIBERO: the standard variant's budget counts macro moves (100), the upstream
  variant's counts control ticks (OpenVLA's 220 on libero-spatial) — the
  package's one documented bend of "task semantics do not change between
  variants" (a language agent cannot emit 20 Hz OSC deltas).
- RoboTwin: the same budget-unit bend (100 macro moves standard, the task's own
  400–1700 action cap upstream), plus two of the port's own, both declared in
  `benchmarks/robotwin.py`: (a) an episode owns a 50-seed window and takes the
  first settled seed in it, where RoboTwin's evaluator draws settled seeds from
  one shared queue — the window makes an episode index mean one scene on every
  run, which the queue does not; (b) the instruction is the task's
  `full_description`, not one of RoboTwin's 50 per-episode templates, because
  those carry `{A}` / `{a}` placeholders only its scripted expert can fill and
  49 of the 50 tasks have no placeholder-free template at all.
- LIBERO-PRO: the instruction is the perturbed BDDL's own `(:language ...)`,
  not the fork's `task.language`. The fork derives that from the task
  *filename*, which the perturbation leaves alone, so it would hand the agent
  the unperturbed sentence and erase the `semantic` dimension.
- LIBERO-Plus: the instruction is the base task's sentence with the encoded
  perturbation suffix stripped. The fork's `task.language` leaves it in
  ("… place it on the plate view 0 0 100 2 352 initstate 0") — not an
  instruction, and it tells the agent the camera angle and noise severity.
  The `libero-plus-language` line keeps the fork's own paraphrase, which it
  reads from a real BDDL file.
- LIBERO-Plus `all` is one trial per task (the benchmark's own
  `num_trials_per_task = 1`), not the 50 init states the base lines use;
  `mini` is the first ten tasks of each base suite, 40 episodes.
- RoboCasa: the same bend (100 macro moves against the task's own horizon,
  300–7200 ticks), plus one of its own — RoboCasa draws each of its 50
  evaluation rollouts from the split's scenes at random and prescribes no seed
  list, so the package deals the 50 round-robin over the line's 10 (RoboCasa365)
  or 5 (v0.2) fixed scenes and seeds each with its scenario index. The set of
  scenes, the object split and the count of 50 are upstream's; which rollout
  lands in which kitchen is ours, so an episode id names a reproducible one.
- RoboCasa: `move_base` is a verb the so101 bench does not have, because the
  so101 does not drive. It is the only added verb; the arm verbs keep their
  so101 signatures and the navigation lines' `step` is still refused here.
- CALVIN: the same budget-unit bend (20 macro moves per instruction against
  CALVIN's own 360 control ticks per instruction), but on this line the budget
  stays **per instruction** on both variants rather than per episode — a chain
  whose instructions shared one pool would let an agent spend everything on the
  first and score 1/5 by construction, which is not what CALVIN measures. The
  20 is ours (decided 2026-09-10); the five instructions, the 360 ticks and the
  "stop at the first failure" rule are upstream's.
- CALVIN: one line, `calvin-d`, where the board has two rows (ABC->D and
  ABCD->D). Those name the *training* split; the evaluation is the same in both
  — environment D and the same 1000 sequences, whose initial states are computed
  from a symbolic condition rather than read from the dataset. A zero-shot agent
  has no training split, so a second line would be the same benchmark twice.
- CALVIN: no new verb and no new arm shape, but the `manip` briefing grew a
  chain form and `core.shapes.es.es_goal` now passes a manip goal's `n_goals`
  through instead of flattening it to 1. Every existing briefing renders
  byte-identical (checked against the pre-change `prompts.py`).
- CALVIN: the port serves the two cameras at 256² instead of CALVIN's 200x200
  static and 84x84 gripper, as the other standard manipulation bodies do. Only
  the raster size changes — both cameras have `aspect: 1` and their projection
  comes from `fov` and `aspect` alone, so the view is the release's.
- BEHAVIOR-1K: the same budget-unit bend (150 macro moves standard against the
  challenge's own per-task tick cap, 4 299–52 120 ticks, which BOTH variants
  keep — only the unit the agent's budget counts differs), plus one of the
  port's own, declared in `benchmarks/behavior.py`: the standard variant
  replaces the challenge's two absolute-joint arm controllers with
  `InverseKinematicsController` in `absolute_pose` mode, so that an absolute
  end-effector target is expressible at all. That is not an invention — it is
  one of the four action-space substitutions the challenge's own
  `docs/challenge/evaluation.md` § "Configure Robot Action Space" documents for
  participants, applied through the same `controller_config` hook its evaluator
  uses. The `-upstream` variant keeps the challenge's 23-D joint action
  untouched. Everything else is the release's: the robot, the frequencies, the
  cameras, the 50 tasks, the 10 scored instances per task, the step budget and
  every metric.
- BEHAVIOR-1K: `mini` is the first scored instance of each of the 50 tasks (50
  episodes), not a prefix of the task list — a scene load costs minutes, and a
  prefix would leave 40 activities untouched.
- BEHAVIOR-1K: the macro protocol does not command the robot's torso, and the
  four trunk joints decide where the arms can reach at all. That is a gap, not
  a choice: the challenge's own action space has them, and the `-upstream`
  variant commands them. Until the protocol gains a torso target, the agent can
  reach only the shell around the posture the instance starts in.

## Serve · smoke · run

```bash
# serve (ac-es; the VLNverse lines need the Isaac launcher reachable from this process)
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  EMBODIEDSCORE_ISAAC_PYTHON=<EmbodiedScore-envs>/scripts/isaac_container.sh \
  ~/miniforge3/envs/ac-es/bin/python -m app.server.auto_host \
  --module exp_workspace.bareES.nodeset --class EnvBareEsNodeSet --port 9270

# token-free wiring pass over every profile (agentcanvas env)
python exp_workspace/bareES/fake_agent.py --server-url http://127.0.0.1:9270 --all

# one episode of one task (test_ prefixed — off the board); one file per task under this folder's configs/, harness=<cc|codex|mini> + model=<name> pick the seat (this folder's configs/harness/ and configs/models/models.yaml), effort=max its effort
python runner.py std_hmeqa_es_bareES harness=cc model=fable-5 run.episodes=0 run.servers=[http://127.0.0.1:9270]
# workers.sh run <group> launches one runner.py per task against its own server (parallel)

# parallel (user layout 2026-09-08): the habitat lines as many workers as profiles —
# one server per profile (:9271-9283) and one runner.py each; the VLNverse lines their own
# group (:9291-9292, Isaac render worker per server). Within a line: ONE worker.
exp_workspace/bareES/workers.sh serve habitat            # 13 servers, waits for /health
exp_workspace/bareES/workers.sh run   habitat            # 13 runners in parallel (add --episodes 0 for a probe)
exp_workspace/bareES/workers.sh serve vlnverse           # needs the EMBODIEDSCORE_ISAAC_* env (see the script header)
exp_workspace/bareES/workers.sh run   vlnverse
exp_workspace/bareES/workers.sh status                   # servers / runs / last episode line per task
exp_workspace/bareES/workers.sh stop  all                # kills the servers it started, by PID
```
Logs and pids: `outputs/logs/bareES/`.

## Status

- 2026-09-08: `fake_agent --all` 15 / 15 profiles green (188 checks); GOAT
  image-goal render verified (val_unseen idx 1, sub-goal 3 "microwave",
  512²). One-episode `cc · fable-5 · default` runs per line: see the
  `test_*_bareES` dirs under `outputs/claudecode/`.
- 2026-09-10: `calvin-d` added (chain of five instructions, port 9330,
  `ac-calvin`). Scripted proof `scripts/calvin_scripted_episode.py --episode 0`
  -> chain_length 2 / 5 in 19 macro steps; first ten sequences
  2 0 1 2 2 0 2 1 1. `+run.fake=true` green on 9330, and on 9331 for
  `std_libero_spatial_es_bareES` (the shared briefing / route / bridge edits do
  not touch the existing manipulation lines).

