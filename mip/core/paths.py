"""Repository paths shared by the runner, the recorders and the monitor.

Nothing here composes a config: the four-layer experiment config
(mip/configs/experiments/<run name>.yaml — task · env · agent · run) is loaded by Hydra in runner.py;
this module only knows where the tree lives.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODING_AGENT = REPO_ROOT  # the runner lives at the repo root
# the configuration every experiment shares (the harness, models and api groups); an experiment
# folder's own copy of a group wins over it (runner.compose_app puts this dir on Hydra's search path)
CONFIGS = REPO_ROOT / "mip" / "configs"
EXPERIMENTS = CONFIGS / "experiments"  # the experiment files of the tool sets (mip.core.toolset)
OUTPUTS = REPO_ROOT / "outputs"

# Evaluation splits (rand100, mip{N}, heldout100 …) are not this repo's: they
# ship with the embodiedscore_envs package
# (embodiedscore_envs/splits/, an overlay of EMBODIEDSCORE_DATA_ROOT, with the
# audit manifests and the generators under its scripts/). An experiment names a
# split; the package's loader resolves it.

# harness name root (agent.harness.root) → output root, also read by the
# monitor (ui/server.py); ``run.dir`` in every experiment file is
# ``${outputs}/<root dir>/${run.name}`` and must agree with this table.
OUTPUT_ROOTS = {
    "cc": OUTPUTS / "claudecode",
    "mini": OUTPUTS / "mini-swe-agent",
    "codex": OUTPUTS / "codex",
    "fake": OUTPUTS / "fake",
}
# the same table by harness key, for the monitor and the backend
OUTPUT_ROOTS_BY_HARNESS = {
    "claudecode": OUTPUT_ROOTS["cc"],
    "mini-swe-agent": OUTPUT_ROOTS["mini"],
    "codex": OUTPUT_ROOTS["codex"],
}
