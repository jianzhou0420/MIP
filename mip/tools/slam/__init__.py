"""The SLAM tools (the paper's SLAM and Hybrid + SLAM interfaces): an onboard ORB-SLAM3 map
the agent can read for free.

    hook.py          SlamHook on EmbodiedScoreEnv, in the env server: the fed micro-stepping
                     world (world/), the instruments' reads as env verbs
    get_pose · get_map · get_trajectory · look_around · mark · preview_path
                     the six instruments, model side (SLAM estimates, never ground truth)
    notes.py         the slam_note a step / goto result carries on a tracking transition

A set with the instruments: ``[nav.observe, nav.step, slam.get_pose, … slam.preview_path]``,
with ``typed_results`` (the SLAM sets' step / goto return dicts) and the env block's hook.

What the family needs from the env server is said here, once: the tool set's ``instruments``
switch reaches the hook's configure (``ENV_FIELDS``, pushed by the run loop), and ``preflight``
refuses a run whose env block carries no SlamHook.
"""

from __future__ import annotations

from typing import Any

#: the tool-set switches the env server's SlamHook is configured with
ENV_FIELDS = ("instruments",)


def preflight(*, env: dict[str, Any], **_facts: Any) -> None:
    """Before episode 0 (mip.core.toolset.ToolSet.preflight): the instruments are served by a
    SlamHook in the env server — without one every slam tool would refuse at its first call."""
    import importlib

    from mip.tools.slam.hook import SlamHook

    for spec in env.get("hooks") or []:
        module, _, name = str(spec.get("_target_") or "").rpartition(".")
        if module and issubclass(getattr(importlib.import_module(module), name), SlamHook):
            return
    raise RuntimeError(
        "the slam tools need a SlamHook in env.hooks (mip.tools.slam.hook.SlamHook) — it serves "
        "their verbs in the env server"
    )
