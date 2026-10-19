# Install

A run is **two interpreters plus the harness CLIs**:

| piece | what | interpreter |
|---|---|---|
| runner | `runner.py`, the framework in `mip/`, the harness SDKs, the MCP bridge | `.conda/coding-agents` (Python 3.10) |
| env server | `EmbodiedScoreEnv`, served in its own process by `mip.core.envserver` — habitat-sim 0.3.3 + `embodiedscore_envs` | `.conda/ac-es` (Python 3.11) |
| harness | Claude Code CLI · Codex CLI · mini-SWE-agent (a Python package, already in the runner env) | node 22, for the two CLIs |

Both envs live inside the clone, under the gitignored `.conda/`, so nothing outside the clone is
consulted and an env of the same name elsewhere does not collide. The name `ac-es` is a contract:
every cell says `env.python: ac-es`, and `mip/core/envserver.py:resolve_python` looks it up in
`<repo>/.conda/ac-es/bin/python` first, then in `<conda root>/envs/ac-es/bin/python` of the
runner's own conda. `env.python=/abs/path/to/python` on the command line overrides both.

## 1. System prerequisites

- x86_64 Linux with an NVIDIA driver (headless EGL rendering: `nvidia-smi` works).
- git, and any conda (the commands below use Miniforge's `mamba`).
- node 22 for the two CLIs (§ 4).
- For the SLAM interfaces only: docker, runnable by your user (§ 6).

## 2. The repo and the runner env: `.conda/coding-agents`

```
git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git   # or `git submodule update --init` after a plain clone
cd MIP
mamba create -y -p .conda/coding-agents python=3.10
.conda/coding-agents/bin/pip install -r requirements.txt   # includes -e thirdparty/EmbodiedScore-envs
```

`conda activate ./.conda/coding-agents` works too; the commands here call the envs by path so
nothing depends on activation.

## 3. The env-server env: `.conda/ac-es`

habitat-sim 0.3.3 comes from the
[EmbodiedScore-habitat](https://github.com/jianzhou0420/EmbodiedScore-habitat) fork, headless,
without bullet. On x86_64 Linux with CPython 3.11 install its wheel:

```
mamba create -y -p .conda/ac-es python=3.11
.conda/ac-es/bin/pip install https://github.com/jianzhou0420/EmbodiedScore-habitat/releases/download/v0.3.3-es.1/habitat_sim-0.3.3-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
.conda/ac-es/bin/pip install -e thirdparty/EmbodiedScore-envs
```

Anywhere else, build it into the env with the fork's `build.sh` (its `BUILD.md` lists the system
packages; it needs CMake ≥ 3.12 and < 4, ninja, GCC ≥ 9 and the EGL headers, about 10 minutes):

```
mamba create -y -p .conda/ac-es -c conda-forge python=3.11 \
    numpy=1.26.4 quaternion attrs pillow=10.4 scipy numba tqdm \
    gitpython imageio imageio-ffmpeg matplotlib pip ninja
git clone --recursive https://github.com/jianzhou0420/EmbodiedScore-habitat.git ../EmbodiedScore-habitat   # ~2.3 GB with submodules
../EmbodiedScore-habitat/build.sh --python "$PWD/.conda/ac-es/bin/python" --verify   # --verify renders one EGL frame
.conda/ac-es/bin/pip install -e thirdparty/EmbodiedScore-envs
```

Check that the contract resolves to the in-repo interpreter:

```
.conda/coding-agents/bin/python -c "from mip.core.envserver import resolve_python; print(resolve_python('ac-es'))"   # <repo>/.conda/ac-es/bin/python
.conda/ac-es/bin/python -c "import habitat_sim, embodiedscore_envs; print(habitat_sim.__version__)"           # 0.3.3
```

## 4. The harness CLIs

```
curl -sS -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash   # then a new shell
nvm install 22
npm install -g @anthropic-ai/claude-code @openai/codex
claude --version && codex --version
```

Log in once per machine: `claude` (subscription) and `codex login`. The cc harness does not run the
npm `claude`: `claude-agent-sdk` ships its own Claude Code binary and reads the same `~/.claude/`
credentials, so the npm install is what gives you the login. A model newer than the bundled binary
needs `harness.cli_path=<a newer claude binary>`. The codex harness shells out to `codex`.
mini-SWE-agent reads the provider's key from the shell that launches the runner:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DASHSCOPE_API_KEY` (qwen rows) or `OLLAMA_URL` (local rows).
The rows are in `mip/configs/models/models.yaml`.

nvm's loader runs only in interactive shells; under `ssh host cmd`, cron or a fresh tmux pane,
source it first (`export NVM_DIR=~/.nvm; . ~/.nvm/nvm.sh`).

## 5. Data

The env server reads `EMBODIEDSCORE_DATA_ROOT` / `EMBODIEDSCORE_SCENE_ROOT`, else
`<repo>/data/embodiedscore/{datasets,scenes}` — a gitignored directory of symlinks. The evaluation
splits (`rand100`, `mip100`) ship inside EmbodiedScore-envs; link only the corpora and the scenes
the cells you run need:

| cells | `datasets/` | `scenes/` |
|---|---|---|
| `std_mip_r2r_*` | `vlnce/R2R_VLNCE_v1-3_preprocessed/` | `mp3d/` |
| `std_mip_rxr_minimal` | `vlnce/RxR_VLNCE_v0/` | `mp3d/` |
| `std_mip_hm3d_minimal` · `std_mip_hm3d_v2_minimal` | `objectnav/hm3d/` | `hm3d/`, `hm3d_v0.2/` |
| `std_mip_mp3d_minimal` | `objectnav/mp3d/` | `mp3d/` |
| `std_mip_hmeqa_minimal` | `hmeqa/` | `hm3dsem/` |

```
mkdir -p data/embodiedscore/{datasets,scenes}
ln -s /path/to/vlnce data/embodiedscore/datasets/vlnce   # holds R2R_VLNCE_v1-3_preprocessed/, RxR_VLNCE_v0/
ln -s /path/to/mp3d  data/embodiedscore/scenes/mp3d      # <scan>/<scan>.glb + .navmesh
```

The exact files each loader reads are in its module docstring
(`thirdparty/EmbodiedScore-envs/embodiedscore_envs/benchmarks/{vlnce,objectnav,hmeqa}.py`). MP3D
and HM3D are licensed by Matterport and are not redistributed.

Runs land in `<repo>/outputs/`; symlink it to keep runs on another disk.

## 6. Services for the richer interfaces

Minimal needs nothing beyond the env server. The others need:

- **Waypoint, Hybrid, Hybrid + SLAM** — the waypoint predictor, an HTTP service at `run.wp_server`
  (default `http://127.0.0.1:9210`) answering `GET /health` and
  `POST /call/smartway_waypoint__predict`. The runner refuses the run when it does not answer.
- **SLAM, Hybrid + SLAM** — docker and the `agentcanvas/orbslam3` image; the env server starts one
  ORB-SLAM3 container per episode (CPU only).

## 7. Verify, in order

```
PY=.conda/coding-agents/bin/python
$PY runner.py std_mip_r2r_minimal harness=cc model=fable-5 +run.fake=true run.episodes=0   # a scripted agent: env server + data + tools, no tokens
$PY runner.py std_mip_r2r_minimal harness=cc model=fable-5 api=fake run.episodes=0        # the real Claude Code CLI on a scripted endpoint, no model
$PY runner.py std_mip_r2r_minimal harness=cc model=fable-5 run.episodes=0                 # one real episode on the subscription
$PY -m ui.server                                                                           # optional: http://<host>:8130, a read-only page over outputs/
```

Each step needs one more piece than the last: step 1 needs § 2, 3 and 5; step 2 the CLIs of § 4;
step 3 the login. The runs land in `outputs/fake/fake_mip_r2r_…`,
`outputs/claudecode/fakeapi_mip_r2r_…` and `outputs/claudecode/test_mip_r2r_…`, each with
`summary.json`, `episode_0.jsonl`, `live_0/`, `env_server.log` and `stats.html`.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: env.python='ac-es': no interpreter at …` | § 3 not done | § 3, or `env.python=/abs/path/to/python` |
| `No matching distribution` / `not a supported wheel on this platform` for habitat_sim | off the wheel's platform (x86_64 Linux, CPython 3.11) | build it (§ 3, second block) |
| `build.sh`: `need CMake >= 3.12 and < 4` | CMake 4 on PATH | `--cmake /path/to/cmake-3.x` |
| `import habitat_sim` → `CXXABI_1.3.15 not found` or `cannot retrieve OpenGL version` | a habitat build without the env RPATH (upstream's build.sh) | rebuild with the fork's `build.sh` |
| `libEGL.so.1` / `EGL context` errors | no graphics driver; in docker, the graphics capability is missing | install the driver; in docker set `NVIDIA_DRIVER_CAPABILITIES=graphics` |
| cc harness log: `ANTHROPIC_API_KEY was set — removed` | by design: subscription auth | `CODING_AGENT_ALLOW_API_KEY=1` to bill via the API |
| `claude`: `/usr/bin/env: 'node': No such file or directory` | node is not on PATH in this shell | § 4; check `which claude` |
| `codex: command not found` from a script, fine in a terminal | nvm loads only in interactive shells | § 4 last paragraph |
