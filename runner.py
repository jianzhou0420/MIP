"""runner — the single entry point: run one experiment from one fully written config.

    python runner.py <experiment.yaml | run name> [key=value …]
    python runner.py --config-name=<run name> [key=value …]        # Hydra's spelling

An experiment is ONE file, ``mip/configs/experiments/<name>.yaml``, four layers and nothing else,
every value written out:

    task    what is measured: data + the benchmark's own budgets and body
    env     where: the env object (its interpreter, port, body) per task family
    agent   who: tools (the tool set from mip/tools/ + prompts + switches) · harness (adapter + constants) · model (id + API params)
    run     this run: name · dir · episodes · servers · wp_server · retry · callbacks (the recorders)

Nothing is looked up elsewhere: a different value is a different experiment (a copy of the
file under a new name). Along the harness and model axes one file is many seats. The harness
is a Hydra config group, ``mip/configs/harness/{cc,codex,mini}.yaml``: the three interchangeable
loops (Claude SDK, Codex CLI, mini-swe-agent), each over the same MCP bridge, that the file
requires (``defaults: - harness: ???``) and the command line picks (``harness=mini``,
``harness.cost_limit=9`` to touch one of its constants); a file runs the three coding agents
unless it lists its own ``harnesses``. The model is one row of the table
``mip/configs/models/models.yaml`` (model -> harness -> the harness-facing id), picked with
``model=gpt-5.6``. ``effort`` (top-level, default ``default``) is the run's reasoning effort
in the vendor's own word, an explicit parameter of every harness adapter. ``api`` is the
endpoint, a third shared group (``mip/configs/api/{native,fake}.yaml``, default native): fake =
the scripted endpoint, the real harness with no model. All three complete the run name
(``std_mip_r2r_cc_opus-5_default_minimal``, ``std_mip_r2r_mini_gpt-5.6_xhigh_minimal``).
Command-line overrides are dotted paths into the same four layers (``run.episodes=0-9``,
``agent.tools.wp_max_moves=45``, ``task.max_turns=300``). The resolved config is saved as
``<run.dir>/config.yaml``.

    python runner.py std_mip_r2r_minimal harness=cc model=fable-5 run.dry_run=true   # print the resolved config, run nothing
    python runner.py mip/configs/experiments/std_mip_r2r_minimal.yaml harness=cc model=fable-5 run.episodes=0-9   # subset probe -> test_…
    python runner.py std_mip_r2r_minimal harness=cc model=fable-5 run.episodes=3,7 run.resume=true   # fill in the full run
    python runner.py std_mip_r2r_minimal harness=cc model=fable-5 effort=max     # the max-effort seat
    python runner.py std_mip_r2r_minimal harness=mini model=qwen3.5-plus         # same file, the open loop
    python runner.py my/probe.yaml                                            # any experiment file, anywhere
    python runner.py std_mip_hmeqa_minimal harness=cc model=fable-5 +run.fake=true   # token-free wiring check (scripted harness)
    python runner.py std_mip_r2r_minimal harness=codex model=gpt-5.6 api=fake run.episodes=0   # the real harness on the scripted endpoint, no model

The launch is one call: ``run(app)`` applies the naming rules, builds the harness adapter, the
interface (the tool set) and the recorders, and hands them to ``mip.core.run.Runner`` — the run loop
(the env pool, the workers, one episode at a time; its docstring has the flow and the graceful
drain).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException, MissingConfigException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError, MissingMandatoryValue

from mip.core.callbacks import Callback
from mip.core.paths import CONFIGS, EXPERIMENTS, REPO_ROOT
from mip.core.retry import DEFAULT_RETRY, RetryPolicy
from mip.core.run import Runner, parse_episodes, run_cfg

OmegaConf.register_new_resolver("eval", eval, replace=True)

# harness keys that name the run / its output root, not adapter kwargs
HARNESS_META = ("root", "output_dir")
#: the harnesses an experiment runs when it lists none (``harnesses:``): the coding-agent loops
CODING_AGENTS = ("cc", "codex", "mini")

# ---------------------------------------------------------------
# region 0. Tools — config, instantiation


def experiments() -> list[Path]:
    """Every experiment file: mip/configs/experiments/*.yaml."""
    return sorted(EXPERIMENTS.glob("*.yaml"))


def resolve_experiment(token: str) -> Path:
    """The experiment file a token names: a path to a .yaml, or a run name
    found under mip/configs/experiments/."""
    path = Path(token)
    if token.endswith(".yaml") or path.is_file():
        if not path.is_file():
            sys.exit(f"[std] no experiment file {path}")
        return path.resolve()
    hits = [p for p in experiments() if p.stem == token]
    if len(hits) != 1:
        sys.exit(
            f"[std] {token!r}: {'no' if not hits else len(hits)} experiment file(s) under "
            f"{EXPERIMENTS} — pass a path"
        )
    return hits[0]


def compose_app(experiment: Path | str, overrides: list[str]) -> dict[str, Any]:
    """The resolved four-layer config of one experiment file (+ overrides)
    as a plain dict."""
    path = resolve_experiment(str(experiment))
    # the file's folder is the Hydra config dir; the shared groups — harness (`defaults: -
    # harness: ???` + `harness=cc`), models (the table `model=fable-5` picks a row of), api — live
    # in mip/configs/, and the prompts group in mip/prompts/, both on Hydra's search
    # path after the file's own folder, so a group the experiment's folder carries itself wins
    searchpath = f"hydra.searchpath=[file://{CONFIGS},file://{REPO_ROOT / 'mip'}]"
    try:
        with initialize_config_dir(config_dir=str(path.parent), version_base=None):
            cfg: DictConfig = compose(config_name=path.stem, overrides=[searchpath, *overrides])
    except (
        ConfigCompositionException,
        MissingConfigException,
    ) as exc:  # no / unknown harness= or model=
        sys.exit(
            f"[std] {path.name}: {str(exc).splitlines()[0]} — harness=<one of: {', '.join(harness_names(path))}> "
            f"model=<one of: {', '.join(model_names(path))}> api=<one of: {', '.join(api_names(path))}>"
        )
    try:  # the chosen model must have an entry under the chosen harness
        OmegaConf.resolve(cfg)
        app = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except (MissingMandatoryValue, InterpolationKeyError) as exc:
        chosen = str(cfg.harness.root) if "harness" in cfg else None
        models = [m for m in model_names(path) if chosen is None or chosen in model_spec(path, m)]
        sys.exit(
            f"[std] {path.name}: {str(exc).splitlines()[0]} — pass "
            f"model=<one of{' (under ' + chosen + ')' if chosen else ''}: {', '.join(models)}>"
        )
    assert isinstance(app, dict)
    if app["harness"]["root"] not in harness_names(path):
        sys.exit(
            f"[std] {path.name} runs harness=<one of: {', '.join(harness_names(path))}>, "
            f"not {app['harness']['root']}"
        )
    # run.dir is written relative in every experiment file (outputs/<harness dir>/<run name>);
    # relative paths in a config are repo-root relative, never cwd
    run_dir = Path(app["run"]["dir"])
    if not run_dir.is_absolute():
        app["run"]["dir"] = str(REPO_ROOT / run_dir)
    return app


def apply_test_rule(app: dict[str, Any]) -> dict[str, Any]:
    """A partial-episode launch is a probe, not an evaluation run: a ``std_`` run over a
    subset of the task's episodes is named ``test_…`` so it never mixes with the full run.
    Filling in the full run takes BOTH ``run.resume=true`` and a run dir that already
    holds records (intent + evidence)."""
    run = app["run"]
    full = set(parse_episodes(str(app["task"]["episodes"])))
    asked = set(parse_episodes(str(run["episodes"])))
    name: str = run["name"]
    if asked == full or not name.startswith("std_"):
        return app
    run_dir = Path(run["dir"])
    has_board = (run_dir / "summary.json").is_file()
    if run.get("resume"):
        if not has_board:
            sys.exit(
                f"[std] run.resume=true but no full run at {run_dir} — "
                "nothing to fill (drop run.resume for a test_ probe)"
            )
        return app  # filling in the existing full run
    new_name = "test_" + name.removeprefix("std_")
    run["name"] = new_name
    run["dir"] = str(run_dir.with_name(new_name))
    hint = (
        " (the existing full run is untouched — add run.resume=true to fill it)"
        if has_board
        else ""
    )
    print(f"[std] partial episodes -> test run {new_name}{hint}")
    return app


FAKE_HARNESS = {
    "_target_": "mip.harnesses.fake.FakeHarness",
    "root": "fake",
    "output_dir": "fake",
    "script": None,
}


def apply_fake(app: dict[str, Any]) -> dict[str, Any]:
    """``+run.fake=true``: run the experiment through the scripted, token-free
    harness (mip.harnesses.fake) — the whole loop against the tool set's real
    bridge, for wiring / runner checks. The harness block is swapped whole
    (its constants are another adapter's), and the run is named ``fake_…``
    and lands under outputs/fake, apart from evaluation runs."""
    run = app["run"]
    if not run.get("fake"):
        return app
    app["agent"]["harness"] = dict(FAKE_HARNESS)
    app["harness_key"] = "fake"
    name = run["name"]
    for prefix in ("std_", "test_", "ui_", "xapi_"):
        name = name.removeprefix(prefix)
    run["name"] = f"fake_{name}"
    run["dir"] = str(Path(run["dir"]).parent.parent / "fake" / run["name"])
    print(f"[std] fake harness -> run {run['name']}")
    return app


def apply_fake_api(app: dict[str, Any]) -> dict[str, Any]:
    """``api=fake`` (the experiment's api/ group): the REAL harness (Claude
    Code / codex / mini) against the scripted endpoint (mip.core.api.fake_wire)
    instead of a vendor — the adapter, its loop, the tool schemas it
    registers and the bridge all exercised, no tokens. The block itself
    reaches the adapter through ModelSpec.extra (api_gateway: fake +
    gateway_task, optionally gateway_script); this rule only names the run
    ``fakeapi_…`` under the harness's own root, apart from evaluation runs."""
    run = app["run"]
    if (app["agent"].get("api") or {}).get("api_gateway") != "fake":
        return app
    if run.get("fake"):
        sys.exit(
            "[std] run.fake and api=fake are exclusive — the scripted harness "
            "(no adapter) or the scripted endpoint (real adapter), not both"
        )
    name = run["name"]
    for prefix in ("std_", "test_", "ui_", "xapi_"):
        name = name.removeprefix(prefix)
    run["name"] = f"fakeapi_{name}"
    run["dir"] = str(Path(run["dir"]).with_name(run["name"]))
    print(f"[std] fake endpoint -> run {run['name']}")
    return app


def build(app: dict[str, Any]) -> tuple[Any, Any, list[Callback]]:
    """Instantiate the agent's parts from the config: the harness adapter,
    the tool set (``agent.tools``) and the recorders."""
    harness = {k: v for k, v in app["agent"]["harness"].items() if k not in HARNESS_META}
    adapter = instantiate(harness, _convert_="all")
    tool_set = instantiate(app["agent"]["tools"], _convert_="all")
    callbacks: list[Callback] = []
    for entry in app["run"]["callbacks"]:
        cb = instantiate(entry, _convert_="all")
        if not isinstance(cb, Callback):
            raise TypeError(f"{entry.get('_target_')!r} is not a Callback subclass")
        callbacks.append(cb)
    return adapter, tool_set, callbacks


def show(app: dict[str, Any], tool_set: Any) -> None:
    """The resolved config plus the flat run cfg the loop derives from it."""
    view = dict(app)
    view["run_cfg"] = run_cfg(app, tool_set)
    print(yaml.safe_dump(view, sort_keys=False, allow_unicode=True), end="")


# endregion
# ---------------------------------------------------------------
# region 3. Main


def run(app: dict[str, Any]) -> None:
    app = apply_fake_api(apply_fake(apply_test_rule(app)))
    adapter, tool_set, callbacks = build(app)
    if app["run"].get("dry_run"):
        show(app, tool_set)
        return
    run_dir = Path(app["run"]["dir"])
    # run.retry: the rate-limit re-run policy (mip.core.retry); absent = DEFAULT_RETRY
    retry = RetryPolicy(**app["run"]["retry"]) if app["run"].get("retry") else DEFAULT_RETRY
    asyncio.run(Runner(adapter, tool_set, app, callbacks, retry=retry).run(run_dir))


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = list(sys.argv[1:] if argv is None else argv)
    experiment: str | None = None
    overrides: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--config-name="):
            experiment = a.split("=", 1)[1]
        elif a in ("--config-name", "-cn"):
            i += 1
            experiment = args[i]
        elif a in ("-h", "--help"):
            print(__doc__)
            return
        elif experiment is None and "=" not in a:
            experiment = a
        else:
            overrides.append(a)
        i += 1
    if not experiment:
        sys.exit("[std] which experiment? a path or a run name (see --help)")
    run(compose_app(experiment, overrides))


def _group_options(experiment: Path, group: str) -> list[str]:
    """A group's options as Hydra sees them: the file's own folder, then mip/configs/."""
    dirs = (experiment.parent / group, CONFIGS / group)
    return sorted({p.stem for d in dirs for p in d.glob("*.yaml")})


def harness_names(experiment: Path) -> list[str]:
    """The harnesses an experiment runs: the file's own ``harnesses`` list, else the coding
    agents — each one yaml of the harness group (mip/configs/harness/)."""
    with open(experiment) as fh:
        declared = (yaml.safe_load(fh) or {}).get("harnesses") or CODING_AGENTS
    group = _group_options(experiment, "harness")
    return [str(h) for h in declared if str(h) in group]


def api_names(experiment: Path) -> list[str]:
    """The experiment's api group: one yaml per endpoint (mip/configs/api/, or the folder's own)."""
    return _group_options(experiment, "api")


def models_table(experiment: Path) -> dict[str, dict[str, Any]]:
    """The experiment's models table, model -> harness -> spec: the folder's own
    models/models.yaml when it carries one, else mip/configs/models/models.yaml (Hydra's order)."""
    own = experiment.parent / "models" / "models.yaml"
    with open(own if own.is_file() else CONFIGS / "models" / "models.yaml") as fh:
        return yaml.safe_load(fh) or {}


def model_names(experiment: Path) -> list[str]:
    return sorted(models_table(experiment))


def model_spec(experiment: Path, name: str) -> dict[str, Any]:
    return models_table(experiment).get(name) or {}


def seats() -> list[tuple[Path, list[str]]]:
    """Every seat: an experiment file plus the overrides that pick it —
    ``harness=<h> model=<m>`` for every harness the file runs and every row
    of the models table with an entry under it."""
    out: list[tuple[Path, list[str]]] = []
    for path in experiments():
        for h in harness_names(path):
            for m in model_names(path):
                if h in model_spec(path, m):
                    out.append((path, [f"harness={h}", f"model={m}"]))
    return out


# endregion
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()
