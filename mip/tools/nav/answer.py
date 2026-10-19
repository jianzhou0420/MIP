"""answer(text) — the EQA lines' terminal: record the answer (a letter on the multiple-choice
lines) and END the episode; rides the env's submit_answer verb."""

from __future__ import annotations

from typing import Any

from mip.core.toolset import Session
from mip.tools.nav.view import check_task

NAME = "answer"
#: by the placement's task word: the multiple-choice lines
DESCRIPTIONS = {
    "eqa": (
        "Permanently END the episode by answering the multiple-choice question. Pass a "
        'single letter: "A", "B", "C" or "D". This is your one and only answer — call it '
        "once you have seen enough of the building to be confident. An episode that ends "
        "without answer() scores zero."
    ),
}


def describe(slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    return DESCRIPTIONS[task]


def register(mcp: Any, s: Session, description: str) -> None:
    check_task(s)

    def answer(text: str) -> dict[str, Any]:
        s.tool_calls += 1
        if s.episode_over:
            return {"error": f"episode already over ({s.end_reason}); your answer was {s.answer!r}"}
        clean = str(text or "").strip().strip("\"'.()").upper()
        if clean not in ("A", "B", "C", "D"):
            return {"error": f'invalid answer {text!r}; pass a single letter: "A", "B", "C" or "D"'}
        outputs = s.call("submit_answer", answer=clean)
        info = outputs.get("info") or {}
        if "error" in info:
            return {"error": info["error"]}
        s.steps_taken += 1  # submit_answer is the STOP step on the discrete lines
        s.episode_over = True
        s.end_reason = "answer_submitted"
        s.answer = clean
        # steps_taken_total makes this the EventSink's last parsed step result,
        # which is how the driver reads the answer back for evaluate().
        result = {
            "answer": clean,
            "answer_submitted": True,
            "episode_over": True,
            "end_reason": s.end_reason,
            "steps_taken_total": s.steps_taken,
            "message": f"Final answer {clean!r} recorded — the episode is over.",
        }
        s.live_log(result)
        return result

    mcp.tool(description=description)(answer)
