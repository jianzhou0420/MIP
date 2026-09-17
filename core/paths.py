"""Repository paths shared by the runner, the recorders and the monitor.

Nothing here composes a config: the four-layer experiment config
(exp_workspace/<arm>/configs/<run name>.yaml — task · env · agent · run) is loaded by Hydra in
runner.py; this module only knows where the tree lives.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODING_AGENT = REPO_ROOT  # the runner lives at the repo root (was vlnworkspace/coding-agent/)
EXP_WORKSPACE = CODING_AGENT / "exp_workspace"
OUTPUTS = REPO_ROOT / "outputs"

# Derived mip{N} evaluation splits: seed-42 scene-stratified proportional
# samples of an official val, MATERIALIZED at the dataset layer (same form as
# R2R-CE's rand100). Audit manifests: splits/*_n100_seed42.json
# (committed — data/ is gitignored, so the manifest is the ONE tracked record
# of what each mip split contains). Generator: sample_episodes.py (git
# history: vlnworkspace cecd19c). splits/ is the single home for ALL split
# data (2026-08-18): flat *_seed42.json manifests plus r2r/ and rxr/ subdirs
# holding the habitat-format split dirs.
SPLITS_DIR = CODING_AGENT / "splits"

# harness name root (agent.harness.root) → output root. The Monitor's
# SOURCE_ROOTS; ``run.dir`` in every experiment file is
# ``${outputs}/<root dir>/${run.name}`` and must agree with this table.
OUTPUT_ROOTS = {
    "cc": OUTPUTS / "claudecode",
    "mini": OUTPUTS / "mini-swe-agent",
    "codex": OUTPUTS / "codex",
    "fake": OUTPUTS / "fake",
    # NavHarness (core/harnesses/navharness.py): the litellm seat and the
    # stateless-SDK seat share one root — the harness is the same code
    "navh": OUTPUTS / "navharness",
    "navhsdk": OUTPUTS / "navharness",
    # ImagineVLN runner (ImagineVLN/agent/run_mapgpt.py) writes the same
    # summary.json + episode_{i}.jsonl + live_{i}/ layout into its own root;
    # the imagine experiments point run.dir there.
    "imagine": OUTPUTS / "imaginevln",
}
# the same table by harness key, for the monitor and the backend
OUTPUT_ROOTS_BY_HARNESS = {
    "claudecode": OUTPUT_ROOTS["cc"],
    "mini-swe-agent": OUTPUT_ROOTS["mini"],
    "codex": OUTPUT_ROOTS["codex"],
    "imagine": OUTPUT_ROOTS["imagine"],
    "navh": OUTPUT_ROOTS["navh"],
}
