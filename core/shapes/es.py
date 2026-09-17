"""es.* — every EmbodiedScore-envs line through the env_bare_es nodeset.

One placement flow for all of them: reset returns the dataset's task
statement (``task``), the goal IN FORCE (``goal`` — a GOAT sequence's first
sub-goal) and ``n_goals``; the task word the briefing gets is read off that
payload (``goal_of``), exactly as the arm's fake_agent derives it. The
declared shape (es.nav / es.objnav / es.goat / es.eqa / es.express /
es.manip) names the line's expected word; a payload that derives
differently is reported, not silently re-shaped. es.manip (the LIBERO,
RoboTwin, RoboCasa and CALVIN lines, 2026-09-10) is the manipulation surface —
observe / state / move_ee / gripper / stop — scored by the package's own
success predicate; the robot's arms come off the placement payload, so the
same shape serves a one-armed Panda and RoboTwin's two arms, and a manip
payload with n_goals > 1 is a CHAIN (CALVIN's five instructions, revealed one
at a time by the environment itself) rather than a GOAT sequence.

The nodeset holds the answer (answer() rides its submit_answer verb) and
scores it — letter match on the multiple-choice lines; EXPRESS needs the
benchmark's gpt-4o-mini judge over the answer AND the final frame first,
whose "δ, sigma" reply evaluate folds into C/C*/E_path. GOAT tightens the
per-episode caps to N sub-goals x the per-goal 口径, under the profile cap.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import EXPLORE_FIRST_PROMPT, FIRST_PROMPT, Goal, TaskModule


def express_judge(system: str, user: str, rgb_b64: str | None) -> str:
    """EXPRESS-Bench's gpt-4o-mini judge (upstream gpt.py:gpt_4o_mini
    mirror): system + user text + the FINAL camera frame, default sampling
    params like upstream. Driver-side; the agent never sees it. Raises on
    transport failure — the caller records the error and evaluate scores
    the benchmark's own judge_ok=0 zero."""
    from core.api import provider_of

    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    if rgb_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{rgb_b64}"},
            }
        )
    resp = provider_of("gpt-4o-mini").complete(
        "gpt-4o-mini",
        [{"role": "system", "content": system}, {"role": "user", "content": content}],
        num_retries=2,
    )
    return str(resp.choices[0].message.content or "")


async def judge_text_for(module: TaskModule, sink: Any, pred: str) -> str:
    """The judge's reply for ``pred``, or "" (no answer, or the judge failed
    — recorded as a driver_error). The express nodeset exposes the
    judge_prompt / observe_egocentric verbs this reads."""
    import asyncio

    if not pred:
        return ""
    try:
        jp = await module.env.call("judge_prompt", pred_answer=pred)
        frame = await module.env.call("observe")
        return await asyncio.to_thread(
            express_judge, str(jp.get("system") or ""), str(jp.get("user") or ""), frame.get("rgb")
        )
    except Exception as exc:
        sink.emit("driver_error", {"error": f"express judge failed: {exc!r}"})
        return ""


def es_goal(ep: dict[str, Any]) -> tuple[str, str, int]:
    """(task word, briefing statement, sub-goal count) from what the
    env_bare_es reset verb returned. point -> nav (the instruction); object
    -> objnav (the category, dataset underscores rendered as spaces);
    question -> eqa (letters) / express (free text); GOAT -> goat, the FIRST
    sub-goal only — upstream's current-subtask sensor never reveals the
    rest, and an image sub-goal's statement must not name the category (the
    photo rides the tool channel)."""
    goal = ep.get("goal") or {}
    kind = str(goal.get("kind") or "none")
    n_goals = int(ep.get("n_goals") or 1)
    task_text = str(ep.get("task") or "").strip()
    if kind == "manip":
        # a manipulation goal stays manip however many sub-goals it has: CALVIN's episode is a
        # chain of five instructions and must NOT fall into the goat branch below, which reads
        # n_goals > 1 as "a GOAT sequence". The count is carried through so the briefing can
        # say how long the chain is and the caps can scale with it.
        return "manip", task_text, n_goals
    if n_goals > 1 or kind in ("image", "description"):
        if kind == "object":
            stmt = f'find a "{str(goal.get("category") or "").replace("_", " ")}" (any instance counts)'
        elif kind == "description":
            stmt = f'find the object this describes — "{goal.get("description")}"'
        elif kind == "image":
            stmt = (
                "find the specific object shown in the photo attached to your first observe() "
                "(that very instance, not just anything of its kind)"
            )
        else:
            stmt = task_text
        return "goat", stmt, n_goals
    if kind == "object":
        return "objnav", task_text.replace("_", " "), 1
    if kind == "question":
        return ("eqa" if goal.get("choices") else "express"), task_text, 1
    return "nav", task_text, 1


class EsTask(TaskModule):
    """Common behaviour of the es.* shapes; ``task_word`` is the declared
    expectation, the goal carries the derived word."""

    shape = "es.nav"
    task_word = "nav"

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        if not ep.get("episode_id"):
            raise RuntimeError(
                f"{self.env.env_name} reset returned no episode (index {index}) "
                "— episode placement failed?"
            )
        task, statement, n_goals = es_goal(ep)
        if not statement:
            raise RuntimeError(
                f"{self.env.env_name} reset returned an empty task statement "
                f"(episode {index}, goal {ep.get('goal')!r})"
            )
        if task != self.task_word:
            print(
                f"[std] WARNING {self.env.env_name} episode {index}: profile "
                f"declares {self.shape} but reset derives {task!r}"
            )
        return Goal(text=statement, task=task, n_goals=n_goals, ep=ep)

    def n_actions(self, goal: Goal) -> int:
        """The line's action-table size: the frozen ``n_actions`` (4 on the
        Isaac lines — yaw-only worker), else 7 with SUBTASK_STOP on GOAT,
        else the STANDARD 6."""
        declared = self.cfg.get("n_actions")
        if declared:
            return int(declared)
        return 7 if goal.task == "goat" else 6

    def caps_of(self, goal: Goal) -> dict[str, Any]:
        if goal.task != "goat":
            return {}
        caps: dict[str, Any] = {}
        if self.cfg.get("max_turns_per_goal"):
            caps["max_turns"] = min(
                int(self.cfg["max_turns"]), int(self.cfg["max_turns_per_goal"]) * goal.n_goals
            )
        if self.cfg.get("episode_timeout_per_goal"):
            caps["episode_timeout"] = min(
                int(self.cfg["episode_timeout"]),
                int(self.cfg["episode_timeout_per_goal"]) * goal.n_goals,
            )
        return caps

    def briefing_kwargs(self, goal: Goal) -> dict[str, Any]:
        return {**super().briefing_kwargs(goal), "n_actions": self.n_actions(goal)}

    def first_prompt(self, goal: Goal) -> str:
        return EXPLORE_FIRST_PROMPT if goal.task in ("eqa", "express") else FIRST_PROMPT

    def bridge_knobs(self, goal: Goal) -> dict[str, Any]:
        return {"es_task": goal.task, "es_actions": self.n_actions(goal)}

    async def evaluate(self, sink: Any, goal: Goal) -> dict[str, Any]:
        inputs: dict[str, Any] = {"trigger": "driver"}
        if goal.task == "express":
            pred = self.answer_of(sink)
            if pred:
                inputs["judge_text"] = await judge_text_for(self, sink, pred)
        return await self.call_evaluate(sink, inputs)

    def record_extras(self, sink: Any, goal: Goal) -> dict[str, Any]:
        out: dict[str, Any] = {}
        answer = (sink.last_step_result or {}).get("answer")
        if goal.task in ("eqa", "express") and answer:
            out["answer"] = answer
        out.update(super().record_extras(sink, goal))
        return out


class EsNav(EsTask):
    shape = "es.nav"
    task_word = "nav"


class EsObjNav(EsTask):
    shape = "es.objnav"
    task_word = "objnav"


class EsGoat(EsTask):
    shape = "es.goat"
    task_word = "goat"


class EsEqa(EsTask):
    shape = "es.eqa"
    task_word = "eqa"


class EsExpress(EsTask):
    shape = "es.express"
    task_word = "express"


MANIP_FIRST_PROMPT = (
    "Begin. Call observe() first to see the workspace, then state() to read the arm."
)


class EsManip(EsTask):
    """The manipulation lines (LIBERO, RoboTwin, RoboCasa): the bare arm surface —
    observe / state / move_ee / gripper / stop, plus move_base on a mobile
    manipulator — over the package's macro pose protocol. The episode ends on
    stop() or the macro-step budget; success is the package's own predicate
    check, latched driver-side, never shown. ``arms`` is whatever the line's
    manager reports, so the briefing and the bridge name the robot's real arms;
    ``base`` (RoboCasa's place() says so) switches the move_base paragraphs on."""

    shape = "es.manip"
    task_word = "manip"

    def first_prompt(self, goal: Goal) -> str:
        return MANIP_FIRST_PROMPT

    def arms(self, goal: Goal) -> list[str]:
        return [str(a) for a in (goal.ep.get("arms") or ["panda"])]

    def briefing_kwargs(self, goal: Goal) -> dict[str, Any]:
        return {**super().briefing_kwargs(goal), "arms": self.arms(goal), "base": bool(goal.ep.get("base"))}

    def caps_of(self, goal: Goal) -> dict[str, Any]:
        """A chain line's per-goal caps scale with the chain, exactly as GOAT's
        do (``EsTask.caps_of`` keys on the goat word; a manip chain gets the
        same treatment)."""
        if goal.n_goals <= 1:
            return {}
        caps: dict[str, Any] = {}
        if self.cfg.get("max_turns_per_goal"):
            caps["max_turns"] = min(int(self.cfg["max_turns"]),
                                    int(self.cfg["max_turns_per_goal"]) * goal.n_goals)
        if self.cfg.get("episode_timeout_per_goal"):
            caps["episode_timeout"] = min(int(self.cfg["episode_timeout"]),
                                          int(self.cfg["episode_timeout_per_goal"]) * goal.n_goals)
        return caps

    def bridge_knobs(self, goal: Goal) -> dict[str, Any]:
        return {"es_task": "manip", "es_actions": 0, "es_arms": ",".join(self.arms(goal))}
