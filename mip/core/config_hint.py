"""The four-layer experiment config, as a dataclass tree — the IDE hint and
the structured schema for mip/configs/experiments/<run name>.yaml.
One file = one experiment, every
value written out. Nothing here composes or validates at runtime — runner.py
composes with Hydra and hands the runner a plain dict — and it is not a
strict schema either: four blocks are open by design (a tool set's switches, a
benchmark's own task keys, a harness's constants, a model's API params), so
``OmegaConf.structured(AppCfg)`` would reject every real file. Read it as
the field-by-field meaning of the four layers.

    task    what is measured: data + the benchmark's own budgets
    env     where: simulator stack + body deviations
    agent   who: tools (the tool set + prompts + switches) · harness · model
    run     this run: name (assembled) · dir · episodes · servers · recorders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class TaskCfg:
    """One benchmark protocol: the data and the budgets the benchmark itself fixes.
    Extra keys are benchmark-specific (episodes_manifest, n_actions,
    max_budget_usd, tilt_actions, habitat_action_space …)."""

    family: str = MISSING  # nav | objnav | eqa — descriptive; the env's place() says what it placed (task_word)
    dataset: Optional[str] = None  # the env panel's dataset selector (None: single-dataset panels)
    split: str = MISSING
    episodes: str = MISSING  # the full set, "0-99"
    max_turns: int = MISSING  # harness turn cap
    step_budget: int = MISSING  # env action budget
    episode_timeout: int = MISSING  # seconds
    rgb_resolution: Any = (
        None  # the benchmark's sensor (512 | "480x640" | [480, 640]); reaches the env's
    )
    # configure() only when env.env_fields lists it (EmbodiedScoreEnv: -> body.rgb width / height)
    require: list[str] = field(default_factory=list)  # data files that must exist


@dataclass
class EnvCfg:
    """One world: the env object (``_target_`` + its kwargs, e.g. ``variant``),
    served by mip.core.envserver under ``python`` on ``port`` — the runner spawns it
    when ``run.servers`` is empty."""

    _target_: str = MISSING  # envs.<env>.env.<Class>
    python: Optional[str] = None  # interpreter: conda env name or path (None: this one)
    port: int = 9200
    env_fields: list[str] = field(
        default_factory=list
    )  # cfg keys passed to configure() (run-level env switches)
    # the env class's own kwargs beyond these — what its __init__ takes:
    #   EmbodiedScoreEnv: variant, gpu_id, body — Body overrides on the variant's preset
    #     (forward_step_m, turn_deg, tilt_deg, camera_pitch_deg, agent_height_m, rgb: {width, height,
    #     hfov_deg, position}, depth: null …); {} = the preset as is
    #   hooks: [{_target_: mip.tools.slam.hook.SlamHook, slam: {…}}] — EnvHooks run in the env server
    #     (mip/envs/embodiedscore/hook.py); SlamHook's slam = the SLAM stack's knobs
    #     (world.stack.SLAM_KEYS): cell_size_m, map_size_m, max_coarse_steps, frontier_match_m,
    #     start_pitch_deg, map_mode (v1 | v2 | v3 | null = v2 for instruments=2 else v1), mapper (the
    #     map class's kwargs), render (render_annotated_map's), success_distance
    body: dict[str, Any] = field(default_factory=dict)
    hooks: list[dict[str, Any]] = field(
        default_factory=list
    )  # EnvHook specs, run in the env server


@dataclass
class ToolsCfg:
    """The agent's interface as a tool set (mip.core.toolset.ToolSet): the mip/tools/ modules it
    registers, in order, behind mip/bridge.py. Any extra key is a switch the tools read as
    an option (bare / auto_observe / wrist …)."""

    _target_: str = "mip.core.toolset.ToolSet"
    word: str = MISSING  # the {word} segment of run.name
    use: list[str] = MISSING  # mip/tools/<family>/<tool>.py, e.g. [nav.observe, nav.step]
    prompts: dict[str, Any] = (
        MISSING  # the composed prompts group (mip/prompts/), ${prompts}: first_message · system · tools
    )


@dataclass
class HarnessCfg:
    """The adapter class + its constants (constructor kwargs); ``root`` and
    ``output_dir`` name the run and its output root. Every adapter takes
    ``effort`` (what the seat word ``default`` sends: null = nothing, the
    product's own default; a level = a constant). Per adapter: cc —
    permission_mode, strict_mcp_config, max_buffer_size, bridge_connect_timeout_s,
    betas, think_budget, haiku_think_budget, cli_path; codex —
    model_reasoning_summary, tools_approval_mode, project_doc_max_bytes, sandbox
    (on argv: --sandbox), probe_timeout_s, max_buffer_size, cli_path; mini — cost_limit, image_window,
    drop_params, model_kwargs (litellm completion kwargs for every model on the
    seat)."""

    _target_: str = MISSING
    root: str = MISSING  # cc | codex | mini | fake — the {harness} segment of run.name
    output_dir: str = MISSING  # outputs/<output_dir>


@dataclass
class ModelCfg:
    """One model for one harness: the harness-facing id, the API routing
    params beside it (api_base / api_gateway / gateway_route …) and its
    sampling ``params`` (temperature / max_tokens / top_p …) — consumed by
    mini (litellm kwargs, over harness.model_kwargs); cc and codex refuse
    them. Effort is not here — it is the top-level ``effort`` key."""

    id: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryCfg:
    """The rate-limit re-run policy (mip.core.retry.RetryPolicy)."""

    max_attempts: int = 6
    base_backoff_s: float = 30.0  # exponential per attempt
    max_backoff_s: float = 300.0  # per-backoff cap


@dataclass
class RunCfg:
    name: str = MISSING  # assembled: {prefix}_{task}_{env}_{harness root}_{model_key}_{effort}_{word}{suffix}
    dir: str = MISSING
    episodes: str = MISSING  # the indices THIS launch runs (default: the task's full set)
    servers: Optional[list[str]] = None  # attach to these env servers; None: spawn run.workers from env
    workers: int = 1  # +run.workers=N on the command line (not a yaml key): run.servers empty -> spawn N env servers on env.port, env.port+1, …; one worker per server
    wp_server: Optional[str] = None  # the waypoint-predictor server (Waypoint / Hybrid tool sets)
    resume: bool = False
    dry_run: bool = False
    budget_usd: Optional[float] = None  # run-level cost fuse (callbacks.CostLogger)
    note: Optional[str] = None  # OmegaConf structured configs need typing.Optional, not X | None
    retry: RetryCfg = field(
        default_factory=RetryCfg
    )  # rate-limit re-runs; absent = mip.core.retry.DEFAULT_RETRY
    server_timeout_s: float = (
        600.0  # a spawned env server: wait for /health this long (cold scene loads)
    )
    client_timeout_s: float = 600.0  # one env verb (place / step / evaluate …) over HTTP
    callbacks: list[Any] = MISSING  # [{_target_: mip.core.callbacks.<mod>.<Class>, …}]


@dataclass
class ApiCfg:
    """The endpoint the harness talks to — the experiment's api/ group,
    picked with ``api=<native | fake>``; every key lands beside the model id
    in ModelSpec.extra. native: nothing (the vendor's own endpoint). fake:
    the scripted endpoint (mip/core/api/fake_wire.py) — a real harness, no
    model; runner.apply_fake_api gives the run a fakeapi_… name."""

    api_gateway: Optional[str] = None  # null | fake (| litellm, from a models-table row)
    gateway_task: Optional[str] = None  # fake: the task key the script closes on (${task_key})
    gateway_script: Optional[Any] = None  # fake: [[tool, args], …] replacing the default walk


@dataclass
class AgentCfg:
    tools: ToolsCfg = MISSING  # the interface: a tool set
    harness: HarnessCfg = MISSING
    model: ModelCfg = MISSING
    api: ApiCfg = field(default_factory=ApiCfg)


@dataclass
class AppCfg:
    # the five-tuple the run name is made of (recorded; run.name is the identity)
    task_key: str = MISSING
    env_key: str = MISSING
    harness_key: str = (
        MISSING  # derived: ${harness.root} — the harness chosen with `harness=<cc | codex | mini>`
    )
    model: str = MISSING  # the model chosen with `model=<name>`: a row of mip/configs/models/models.yaml
    model_key: str = MISSING  # derived: ${model}
    effort: str = "default"  # reasoning effort, the vendor's own word: Claude low|medium|high|max,
    # OpenAI low|medium|high|xhigh; "default" = the API's own default, nothing sent. An explicit
    # parameter of every harness adapter (EpisodeContext.effort), and a segment of run.name.
    prefix: str = "std"
    suffix: str = ""
    harness: HarnessCfg = MISSING  # the shared Hydra group mip/configs/harness/<name>.yaml (defaults: - harness: ???); agent.harness = ${harness}
    models: dict[str, dict[str, Any]] = (
        MISSING  # mip/configs/models/models.yaml (defaults: - models: models): model -> harness -> ModelCfg; agent.model = ${models[${model}][${harness.root}]}
    )
    task: TaskCfg = MISSING
    env: EnvCfg = MISSING
    agent: AgentCfg = MISSING
    run: RunCfg = MISSING
