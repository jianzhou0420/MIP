"""Task shapes — ``env.shapes.<family>`` → TaskModule class.

One stack (EmbodiedScore-envs). env_bare_es speaks the unified ES reset /
evaluate and takes the es.* shapes (EsNav · EsObjNav · EsGoat · EsEqa ·
EsExpress · EsManip); env_slam_es keeps the slam panel's per-family verbs and takes
nav.Nav · objnav.ObjNav · eqa.Eqa. runner.py resolves the class with
hydra.utils.get_class and instantiates it per episode with (env, cfg, arm).
See core.shapes.base for the contract.
"""

from __future__ import annotations

from core.shapes.base import Goal, Placement, TaskModule, format_choices
from core.shapes.eqa import Eqa
from core.shapes.es import (
    EsEqa,
    EsExpress,
    EsGoat,
    EsManip,
    EsNav,
    EsObjNav,
    es_goal,
    express_judge,
)
from core.shapes.manip import Manip
from core.shapes.nav import Nav
from core.shapes.objnav import ObjNav

SHAPES: dict[str, type[TaskModule]] = {
    cls.shape: cls
    for cls in (Nav, ObjNav, Eqa, EsNav, EsObjNav, EsGoat, EsEqa, EsExpress, EsManip, Manip)
}

__all__ = [
    "SHAPES",
    "Goal",
    "Placement",
    "TaskModule",
    "es_goal",
    "express_judge",
    "format_choices",
]
