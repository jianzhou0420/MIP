"""eqa — multiple-choice embodied QA (HM-EQA / MT-HM3D), end by answer(letter).

Serves env_slam_es's hmeqa profiles. reset's ``question`` port is the RAW
text, so the formatted multiple-choice question
the briefing's {question} slot needs is built HERE from question + choices.
The agent's letter rides the tool-result channel (the bridges' answer()
result carries steps_taken_total, so it lands as the sink's last parsed
step result) and reaches evaluate as ``pred_letter``; an episode that never
answered evaluates "" -> success 0, the benchmark's own zero.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import EXPLORE_FIRST_PROMPT, Goal, TaskModule, format_choices


class Eqa(TaskModule):
    shape = "eqa"
    task_word = "eqa"

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        raw_q = str(ep.get("question") or "").strip()
        choices = ep.get("choices") or []
        if not raw_q or not choices:
            raise RuntimeError(
                f"{self.env.env_name} reset returned question={raw_q!r} "
                f"choices={choices!r} (episode {index}) — episode placement "
                "failed, or the env is not on dataset=hmeqa?"
            )
        return Goal(text=format_choices(raw_q, choices), task="eqa", ep=ep)

    def first_prompt(self, goal: Goal) -> str:
        return EXPLORE_FIRST_PROMPT

    def bridge_knobs(self, goal: Goal) -> dict[str, Any]:
        # the slam arms' task surface + their HM-EQA-only tilt mask (their
        # bridges gate it behind the eqa task; None keeps it out of the env)
        return {"slam_task": "eqa", "slam_tilt": bool(self.cfg.get("tilt_actions", True))}

    async def evaluate(self, sink: Any, goal: Goal) -> dict[str, Any]:
        return await self.call_evaluate(sink, {"pred_letter": self.answer_of(sink)})

    def record_extras(self, sink: Any, goal: Goal) -> dict[str, Any]:
        out: dict[str, Any] = {}
        answer = (sink.last_step_result or {}).get("answer")
        if answer:
            out["answer"] = answer
        out.update(super().record_extras(sink, goal))
        return out
