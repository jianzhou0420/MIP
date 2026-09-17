"""manip — open-ended manipulation on a real arm (the so101 env).

The env's ``place`` returns the task statement verbatim (``task``), an
``episode_id``, the arms the episode uses (``arms``) and a ``goal`` of kind
``manip``. Nothing is scored: the line has no ruler yet (user decision
2026-09-10 — wire the arms in first, judge later), so ``evaluate`` records
the env's end-of-episode facts (steps, end reason, final joint state) and
``success`` stays null. The episode ends when the agent calls stop() or the
step budget runs out.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import Goal, TaskModule

MANIP_FIRST_PROMPT = (
    "Begin. Call observe() first to see the workspace, then state() to read the arms."
)


class Manip(TaskModule):
    shape = "manip"
    task_word = "manip"

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        text = str(ep.get("task") or "").strip()
        if not text:
            raise RuntimeError(
                f"{self.env.url} place({index}) returned no task statement — episode placement failed?"
            )
        return Goal(text=text, task="manip", n_goals=1, ep=ep)

    def first_prompt(self, goal: Goal) -> str:
        return MANIP_FIRST_PROMPT

    def briefing_kwargs(self, goal: Goal) -> dict[str, Any]:
        return {**super().briefing_kwargs(goal), "arms": list(goal.ep.get("arms") or [])}

    async def evaluate(self, sink: Any, goal: Goal) -> dict[str, Any]:
        return await self.call_evaluate(sink, {"trigger": "driver"})
