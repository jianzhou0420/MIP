"""EnvHook — an extension of EmbodiedScoreEnv that runs inside the env server's process, for a
tool that must see every step the world takes (SLAM). The experiment file attaches it to the
env block::

    env:
      _target_: mip.envs.embodiedscore.env.EmbodiedScoreEnv
      hooks:
      - _target_: mip.tools.slam.hook.SlamHook
        slam: {...}

A hook may build the line's gym stack itself (to put a layer under the metric wrapper), act
around each episode's reset, add keys to every step's info and to the metrics, take configure()
fields, and serve verbs of its own or in place of the env's (on a navigation line a hook's verb
replaces the env's own of the same name, any name — ``EmbodiedScoreEnv._serve``; a verb only a
hook serves is reached by name). It reaches the episode through the manager's hook surface
(``EmbodiedScoreEnvManager``: lock, stack, base, ensure_placed, advance, step_info, refresh_view,
previous_info, dataset_index …).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class EnvHook:
    #: configure() kwargs this hook takes (the env block's env_fields push them)
    fields: tuple[str, ...] = ()

    def bind(self, manager: Any) -> None:
        self.mgr = manager

    def configure(self, **fields: Any) -> None:
        """The run's values of ``fields`` (only those the call carried)."""

    def make_stack(
        self, line: str, split: str, *, data_root: str, scene_root: str, gpu_id: int
    ) -> Any | None:
        """The line's gym stack, or None for the env's own (``es.make``)."""
        return None

    def around_reset(self, index: int, reset: Callable[[], tuple[dict, dict]]) -> tuple[dict, dict]:
        """Seat run index ``index``: ``reset()`` is the line's reset (through the hooks after
        this one); a hook may act before and after it. Returns its (obs, info)."""
        return reset()

    def step_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Keys added to a step's info, from the last gym step's ``info``."""
        return {}

    def metrics(self, metrics: dict[str, Any]) -> None:
        """Add to (or rewrite) the episode's metrics in place."""

    def verbs(self) -> dict[str, Callable[..., dict]]:
        """Env-server verbs this hook serves, by name; a name the env also serves is replaced."""
        return {}
