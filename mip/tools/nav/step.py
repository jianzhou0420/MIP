"""step(actions): run a list of discrete actions in order, stopping early when the episode ends.

The action table is the benchmark's (``task.n_actions``). The new view is returned only when the
``auto_observe`` switch is on; otherwise the agent calls observe() to look. With
``typed_results`` (the Hybrid set) the tool returns a dict, which FastMCP also sends as
structuredContent."""

from __future__ import annotations

from typing import Any

from mip.core.toolset import Session
from mip.tools.nav.view import action_error, check_task, execute_actions, over_error, with_view

NAME = "step"
HEAD = "Execute a sequence of movement actions, in order."
TAIL = (
    "Executes sequentially and halts early {halts}. Returns how many actions "
    "ran, total steps taken, remaining budget, and whether the episode is over. The "
    "camera view changes after stepping — call observe() to see the result."
)
#: slots: ``actions`` = the action table (from the prompts group's tools.step.actions);
#: ``halts`` (optional) = when the action list stops early
DESCRIPTION = HEAD + "\n\nActions: {actions}.\n\n" + TAIL
HALTS = "if the episode ends"


def describe(slots: dict[str, Any], *, task: str, names: tuple[str, ...]) -> str:
    return DESCRIPTION.format(**{"halts": HALTS, **slots})


def register(mcp: Any, s: Session, description: str) -> None:
    check_task(s)
    auto_observe = bool(s.options.get("auto_observe"))

    def run(actions: list[int]) -> Any:
        s.tool_calls += 1
        error = over_error(s) or (
            {"error": "empty action list"} if not actions else action_error(s, actions)
        )
        if error:
            return error
        return with_view(s, execute_actions(s, actions), auto_observe=auto_observe)

    if s.options.get("typed_results"):

        def step(actions: list[int]) -> dict[str, Any]:
            return run(actions)

    else:

        def step(actions: list[int]) -> list:  # a bare `list` return type: FastMCP sends unstructured content only
            return run(actions)

    mcp.tool(description=description)(step)
