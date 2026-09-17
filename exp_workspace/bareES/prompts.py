"""bareES — the bare briefing on every EmbodiedScore-envs line (exp_workspace contract).

One builder, six task shapes. The nav text is bare's frozen briefing
(``exp_workspace/bare/prompts.py``) with the STANDARD action table (tilt
actions 4/5 added — user decision 2026-09-08: the STANDARD table on every
line); objnav / eqa / express are the objnav / hmeqa / express arms' frozen
briefings re-said on the two-tool observe/step surface with the STANDARD
magnitudes (15° turns, 30° tilt); goat is new (GOAT-Bench has no bare-arm
precedent). The builder bakes the arm's knobs — the driver passes only what
the signature names: ``task`` (nav / objnav / goat / eqa / express),
``n_actions`` (4 on the Isaac lines — the worker renders yaw-only poses —
else 6), ``n_goals`` (GOAT sub-goal count). manip (2026-09-10) is the LIBERO lines'
briefing on the arm surface (observe / state / move_ee / gripper / stop) —
the so101 bench's text re-said for the simulated Panda and the world frame.

Classic observe/step alternation on every line (auto_observe OFF), as bare.
"""

from __future__ import annotations

_TOOLS_HEAD = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. {actions}
"""

_ACTIONS_MOVE = "1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees"
_ACTIONS_TILT = (
    ", 4 = tilt the camera up 30 degrees, 5 = tilt the camera down 30 degrees "
    "(tilt changes the camera pitch only, not your position or heading)"
)
_TILT_RULE = (
    "\n- Tilt changes persist until you change them again; level the camera again "
    "before moving on, and check your camera pitch before interpreting a view."
)

NAV_BODY = """\

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction \
wants you to go next, move, look again.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.{tilt_rule}
- Work autonomously until you stop; nobody can answer questions.
"""

OBJNAV_BODY = """\

Your task is object-goal navigation: no route is given — search the \
building until you find the target object, walk up to it, and stop there.

Target object: "{instruction}"

Rules:
- Alternate observing and stepping: look, decide where to search next, \
move, look again.
- Explore efficiently: sweep toward where a "{instruction}" is most likely to \
be, and avoid re-walking areas you have already ruled out.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while standing right next \
to a "{instruction}" (within about 1 meter). Any instance counts. STOP is \
permanent — issue it only when you can see the target and are standing \
right next to it.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around \
when unsure.{tilt_rule}
- Work autonomously until you stop; nobody can answer questions.
"""

GOAT_BODY = """\

Your task is lifelong multi-object navigation (GOAT-Bench): this episode has \
{n_goals} sub-goals in this same building, to be reached ONE AT A TIME, in \
order. Each sub-goal is given in one of three ways — an object category \
(any instance counts), a language description of one specific object, or a \
photo of one specific object. You only ever see the CURRENT sub-goal; the \
next one is announced after you close the current one. Targets can repeat \
later in the episode, so remember where things are.

Sub-goal 1 of {n_goals}: {instruction}

Rules:
- Alternate observing and stepping: look, decide where to search next, \
move, look again.
- When you are standing right next to the current target (within 0.25 m), \
issue action 6 (SUBTASK_STOP) to close it. The step result then announces \
the next sub-goal — read it before moving on; when it is a photo, the photo \
is attached to that result (and the first sub-goal's photo, when it is one, \
is attached to your first observe()).
- Closing a sub-goal is permanent: you are scored on where you stood when \
you closed it, whether or not you had found it. Closing the last sub-goal \
ends the episode. Action 0 (STOP) also ends the whole episode — you should \
not need it.
- You have a budget of {budget} movement actions for the whole episode.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around \
when unsure.{tilt_rule}
- Work autonomously until the episode ends; nobody can answer questions.
"""

EQA_BODY = """\
- answer(letter): permanently END the episode by answering the question \
with "A", "B", "C" or "D".

Your task is embodied question answering: explore the building until you \
can answer this multiple-choice question about it:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where to go to find the \
evidence the question needs, move, look again.
- You have a budget of {budget} movement actions. If it runs out you can \
still observe and answer from where you stand.
- You succeed only if answer() gives the correct letter. answer() is \
permanent — call it once you have seen enough evidence to be confident, \
and always answer before ending: an episode without answer() scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.{tilt_rule}
- Work autonomously until you answer; nobody can help you.
"""

EXPRESS_BODY = """\
- answer(text): permanently END the episode by answering the question in \
free-form natural language (one or two sentences).

Your task is embodied question answering: explore the building until you \
can answer this question about it:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where to go to find the \
evidence the question needs, move, look again.
- You have a budget of {budget} movement actions. If it runs out you can \
still observe and answer from where you stand.
- Your answer is judged on BOTH its correctness and whether your final \
camera view supports it — walk up to the relevant object or place and \
answer while it is in view.
- answer() is permanent — call it once you have seen enough evidence to be \
confident, and always answer before ending: an episode without answer() \
scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.{tilt_rule}
- Work autonomously until you answer; nobody can help you.
"""

MANIP_BASE_TOOL = """\
- move_base(dx, dy, dyaw): drive the mobile base by a displacement in its own frame — dx metres \
forward (negative = backwards), dy metres to the robot's left (negative = right), dyaw degrees \
counter-clockwise. {arms_hold} while the base moves.
"""

MANIP_BASE_RULE = """\
- The robot drives: {arms_reach} reaches roughly 0.8 m around the base, so if move_ee reports \
reached=false the target is probably out of reach from where you stand — move_base toward it (0.2-0.5 m \
at a time), re-observe, then reach again. state() reports the base pose as well as the {grippers}.
"""

MANIP_CHAIN_TASK = """\
This task is a SEQUENCE of {n_goals} instructions, given to you ONE AT A TIME. You are told the \
one you are working on now; the rest are not shown and cannot be worked on ahead.

INSTRUCTION 1 of {n_goals}: {instruction}

How the sequence runs:
- The environment checks the workspace after every move and decides for itself when the current \
instruction is done — there is no tool for declaring it. The moment it is satisfied, the result of \
whatever move finished it says so and gives you the NEW INSTRUCTION. Read every move result.
- Each instruction gets its own budget of {budget} motion calls, and the budget starts again with \
each new instruction. If one runs out before the instruction is satisfied, the whole task ends \
there — the remaining instructions are never given. So work the current one deliberately and \
finish it.
- Nothing is undone between instructions: the workspace carries over exactly as you left it.\
"""

MANIP_HEAD = """\
You are controlling {robot} in a simulated {workspace}{mount}, seen by a fixed third-person \
camera{wrist}. You act through tools; plan before you move.

{task_block}

Tools:
- observe(): the third-person camera's current frame{wrist_tool}. Free — use it often.
- state(): for {arms_phrase}, the gripper's position (x, y, z, metres) and orientation (roll, pitch, \
yaw, degrees) in the world frame, and the gripper opening (0 = closed .. 100 = open); plus the robot \
base position. Free.
- move_ee(arm, x, y, z, roll, pitch, yaw, gripper): move one gripper to an absolute target position in \
the world frame (metres; z up; {table_note} — read state() for the exact numbers). Orientation is \
optional (all three angles or none; none keeps the current one). gripper is optional (0-100). \
{arms_clause} The call returns the pose actually reached; reached=false with a position error means the \
target was out of reach or blocked.{other_arm_note}
{base_tool}- gripper(arm, open): open (100) or close (0) that gripper and let it settle. Closing on an object grasps it.
- stop(): declare the task done and END the episode.

Every motion call blocks until the arm settles and returns the pose reached and the budget used.

How to work:
- Look first (observe), read the arms (state), then move in a few deliberate steps and re-observe after \
each move to see the effect. The cameras are your only feedback about the objects.
{base_rule}- To pick something up: move above it with the gripper open, descend so the fingers straddle it (a bowl \
or a plate is grasped by its rim, a small object around its body), close the gripper, then lift. To place: \
move above the destination, lower, open the gripper, lift away.
{reach_rule}\
{budget_rule}\
Call stop() when the task is done or you cannot progress further.
"""

MANIP_BUDGET_RULE = """\
- You have {budget} motion calls in total ({motion_verbs} each count one); the episode is truncated \
when the budget is spent. \
"""

MANIP_CHAIN_BUDGET_RULE = """\
- You have {budget} motion calls for the CURRENT instruction ({motion_verbs} each count one); each \
move result says how many are left on it, and the count starts again when a new instruction arrives. \
"""

# The robot's shape: one fixed-base arm (LIBERO), two arms (RoboTwin), one arm on a mobile base
# (RoboCasa), two arms on a mobile base (BEHAVIOR-1K's R1 Pro), or one arm on CALVIN's play table.
# ``base`` picks the mobile wording of whichever arm count the line reports; the chain flag
# (n_goals > 1) picks the play table for the one-arm shape.
_MANIP_ONE_ARM = {
    "robot": "a Franka Panda robot arm with a parallel-jaw gripper",
    "arms_phrase": "the arm",
    "arms_clause": 'arm is "panda".',
    "table_note": "the table top is at about z = 0.88 and the robot base sits at the far side from the camera",
    "other_arm_note": "",
    "reach_rule": (
        "- The gripper opens about 8 cm; approach objects from above with the fingers pointing down, and "
        "turn the wrist (yaw) so the jaws straddle the part you want to grip.\n"
    ),
}

_MANIP_TWO_ARMS = {
    "robot": "a two-armed robot with a parallel-jaw gripper on each arm",
    "arms_phrase": "each arm",
    "arms_clause": 'arm is "left" or "right".',
    "table_note": (
        "the table top is at about z = 0.75, x runs left-to-right across it and y from the robot toward "
        "the camera"
    ),
    "other_arm_note": " The other arm holds its pose and its grip while this one moves.",
    "reach_rule": (
        "- Each arm reaches its own half of the table best: use the left arm for objects on the left "
        "(negative x) and the right arm for objects on the right. Approach from above with the fingers "
        "pointing down, and turn the wrist (yaw) so the jaws straddle the part you want to grip. Some "
        "tasks need both arms — one to hold, one to act, or one on each side of a wide object.\n"
    ),
}

_MANIP_MOBILE = {  # RoboCasa: the one-arm shape on a wheeled base in a kitchen
    **_MANIP_ONE_ARM,
    "table_note": "the kitchen counters are at about z = 0.9 and the robot drives on the floor",
}

_MANIP_PLAY_TABLE = {  # CALVIN: the one-arm shape on the play table (picked by the chain flag)
    **_MANIP_ONE_ARM,
    "table_note": (
        "the table top is at about z = 0.46, x runs left-to-right across it (positive x is to the "
        "right as the camera sees it) and y from the front edge toward the back, where the sliding "
        "cabinet, the drawer, the button and the switch are"
    ),
    "reach_rule": (
        "- The gripper opens about 8 cm; approach objects from above with the fingers pointing down, "
        "and turn the wrist (yaw) so the jaws straddle the part you want to grip. The drawer and the "
        "sliding door are pulled and pushed by their handles, the switch is flipped and the button "
        "pressed with the closed gripper.\n"
    ),
}

_MANIP_TWO_ARMS_MOBILE = {  # BEHAVIOR-1K: the two-arm shape on a wheeled base, in a whole house
    **_MANIP_TWO_ARMS,
    "table_note": (
        "counters and tables are at about z = 0.8-0.9, shelves and cabinets higher, the floor at "
        "z = 0, and the robot drives from room to room"
    ),
    "reach_rule": (
        "- Each arm reaches its own side best: use the left arm for what is on your left and the "
        "right arm for what is on your right, and both when an object is wide or must be held "
        "while the other hand works. Approach from above with the fingers pointing down, and turn "
        "the wrist (yaw) so the jaws straddle the part you want to grip.\n"
    ),
}


_BODIES = {
    "nav": NAV_BODY,
    "objnav": OBJNAV_BODY,
    "goat": GOAT_BODY,
    "eqa": EQA_BODY,
    "express": EXPRESS_BODY,
}


def _actions_clause(task: str, n_actions: int) -> str:
    tilt = _ACTIONS_TILT if n_actions >= 6 else ""
    if task in ("eqa", "express"):
        return f"{_ACTIONS_MOVE}{tilt}. There is no stop action."
    stop = {
        "nav": "0 = STOP (permanently ends the episode — declares you have reached the goal), ",
        "objnav": "0 = STOP (permanently ends the episode — declares you have found the target), ",
        "goat": "0 = STOP (permanently ends the whole episode), ",
    }[task]
    goat = (
        ", 6 = SUBTASK_STOP (declare the current sub-goal reached and move on to the next)"
        if task == "goat"
        else ""
    )
    return f"{stop}{_ACTIONS_MOVE}{tilt}{goat}."


def build_briefing(
    instruction: str,
    step_budget: int,
    task: str = "nav",
    n_actions: int = 6,
    n_goals: int = 1,
    arms: list[str] | None = None,
    base: bool = False,
) -> str:
    if task == "manip":
        # the driver passes the arms only on the manip lines; the wrist views ride observe().
        # ``base``: the robot drives (RoboCasa's place() says so) — the move_base tool and its rule.
        names = [str(a) for a in (arms or [])]
        wrist = bool(names)
        two = len(names) > 1
        if two:
            shape = _MANIP_TWO_ARMS_MOBILE if base else _MANIP_TWO_ARMS
        else:
            shape = (
                _MANIP_MOBILE if base else (_MANIP_PLAY_TABLE if n_goals > 1 else _MANIP_ONE_ARM)
            )
        cameras = "a wrist camera on each gripper" if two else "a wrist camera on the gripper"
        views = "the wrist cameras' views" if two else "the wrist camera's view"
        # a manipulation task with more than one goal is a CHAIN (CALVIN): the instructions
        # arrive one at a time and the budget is per instruction, not per episode.
        chain = n_goals > 1
        motion_verbs = "move_ee, move_base and gripper" if base else "move_ee and gripper"
        task_block = (
            MANIP_CHAIN_TASK.format(
                instruction=instruction.strip(), n_goals=n_goals, budget=step_budget
            )
            if chain
            else f"TASK: {instruction.strip()}"
        )
        budget_rule = (MANIP_CHAIN_BUDGET_RULE if chain else MANIP_BUDGET_RULE).format(
            budget=step_budget, motion_verbs=motion_verbs
        )
        return MANIP_HEAD.format(
            task_block=task_block,
            budget_rule=budget_rule,
            budget=step_budget,
            workspace=("house" if two else "kitchen") if base else "tabletop workspace",
            mount=" on a wheeled mobile base" if base else "",
            wrist=f" and {cameras}" if wrist else "",
            wrist_tool=f" followed by {views}" if wrist else "",
            motion_verbs=motion_verbs,
            base_tool=MANIP_BASE_TOOL.format(
                arms_hold="Both arms hold their poses" if two else "The arm holds its pose"
            )
            if base
            else "",
            base_rule=MANIP_BASE_RULE.format(
                arms_reach="an arm alone" if two else "the arm alone",
                grippers="grippers'" if two else "gripper's",
            )
            if base
            else "",
            **shape,
        )
    if task not in _BODIES:
        raise ValueError(f"bareES: unknown task {task!r}")
    head = _TOOLS_HEAD.format(actions=_actions_clause(task, n_actions))
    body = _BODIES[task].format(
        instruction=instruction,
        budget=step_budget,
        n_goals=n_goals,
        tilt_rule=_TILT_RULE if n_actions >= 6 else "",
    )
    return head + body
