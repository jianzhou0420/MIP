# Install

A run is **two interpreters plus the harness CLIs**, and a third interpreter for the waypoint interfaces:

| piece | what | interpreter |
|---|---|---|
| runner | `runner.py`, the framework in `mip/`, the harness SDKs, the MCP bridge | `.conda/coding-agents` (Python 3.10) |
| env server | `EmbodiedScoreEnv`, served in its own process by `mip.core.envserver` — habitat-sim 0.3.3 + `embodiedscore_envs` | `.conda/ac-es` (Python 3.11) |
| harness | Claude Code CLI · Codex CLI · mini-SWE-agent (a Python package, already in the runner env) | node 22, for the two CLIs |
| waypoint predictor | the Waypoint / Hybrid interfaces only: `mip/tools/waypoint/predictor/server.py` (§ 6) | `.conda/ac-wp` (Python 3.10) |

The envs live inside the clone, under the gitignored `.conda/`, so nothing outside the clone is
consulted and an env of the same name elsewhere does not collide. The name `ac-es` is a contract:
every cell says `env.python: ac-es`, and `mip/core/envserver.py:resolve_python` looks it up in
`<repo>/.conda/ac-es/bin/python` first, then in `<conda root>/envs/ac-es/bin/python` of the
runner's own conda. `env.python=/abs/path/to/python` on the command line overrides both.

## 1. System prerequisites

- x86_64 Linux with an NVIDIA driver (headless EGL rendering: `nvidia-smi` works).
- git, and any conda (the commands below use Miniforge's `mamba`).
- node 22 for the two CLIs (§ 4).
- For the SLAM interfaces only: docker, runnable by your user (§ 6).
- For the waypoint interfaces only: a CUDA GPU (the predictor's torch is built for CUDA 12.8; tested with driver 570, § 6).

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
.conda/ac-es/bin/pip install pillow==10.4.0
.conda/ac-es/bin/pip install -e thirdparty/EmbodiedScore-envs
```

Pillow is pinned because the panorama the waypoint interfaces render encodes its depth as a 16-bit
PNG through `Image.fromarray(..., mode="I;16")`, which newer Pillow (13) refuses; 10.4.0 is what the
paper's env server ran.

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
`<repo>/data/embodiedscore/{datasets,scenes}` — a gitignored directory of symlinks. The episodes of
every evaluation split (R2R `rand100`, RxR / ObjectNav `mip100`, the HM-EQA question draw) ship
inside EmbodiedScore-envs, so most cells need only the scenes:

| cells | `datasets/` | `scenes/` |
|---|---|---|
| `std_mip_r2r_*` · `std_mip_rxr_minimal` · `std_mip_mp3d_minimal` | — | `mp3d/` |
| `std_mip_hm3d_minimal` | — | `hm3d/` |
| `std_mip_hm3d_v2_minimal` | — | `hm3d_v0.2/` |
| `std_mip_hmeqa_minimal` | `hmeqa/` (`questions.csv`, `scene_init_poses.csv`) | `hm3dsem/` |

```
mkdir -p data/embodiedscore/{datasets,scenes}
ln -s /path/to/mp3d data/embodiedscore/scenes/mp3d   # <scan>/<scan>.glb + .navmesh
ln -s /path/to/hm3d data/embodiedscore/scenes/hm3d   # val/<id>/<short>.basis.glb + hm3d_annotated_basis.scene_dataset_config.json
```

`hm3d/` needs the annotated scene-dataset config next to `val/`; HM3D v2 episodes name their scenes
`hm3d_v0.2/val/…`, so `hm3d_v0.2/` may be the v0.2 download or a directory whose `val` links to
`hm3d/val`.

The exact files each loader reads are in its module docstring
(`thirdparty/EmbodiedScore-envs/embodiedscore_envs/benchmarks/{vlnce,objectnav,hmeqa}.py`). MP3D
and HM3D are licensed by Matterport and are not redistributed.

Runs land in `<repo>/outputs/`; symlink it to keep runs on another disk.

## 6. Services for the richer interfaces

Minimal needs nothing beyond the env server. The others need:

- **SLAM, Hybrid + SLAM** — docker, runnable by your user, and the `mip/orbslam3` image; the
  env server starts one ORB-SLAM3 container per episode (CPU only). Build the image once (about
  12 minutes at `JOBS=4`; `JOBS` is the compiler parallelism; lower it on a machine short of RAM):

  ```
  docker build --build-arg JOBS=4 -t mip/orbslam3:latest mip/tools/slam/docker
  docker run --rm mip/orbslam3:latest python3 -c "import orbslam3"   # the binding loads
  ```

- **Waypoint, Hybrid, Hybrid + SLAM** — the waypoint predictor, `mip/tools/waypoint/predictor/`:
  the waypoint network of [SmartWay](https://github.com/sxyxs/SmartWay-Code) (vendored, MIT) behind
  HTTP at `run.wp_server` (default `http://127.0.0.1:9210`). It runs in a third env, `.conda/ac-wp`
  (Python 3.10, torch 2.11 for CUDA 12.8), on the GPU (under 1 GB of VRAM), with two checkpoints:

  ```
  mamba create -y -p .conda/ac-wp python=3.10
  .conda/ac-wp/bin/pip install -r mip/tools/waypoint/predictor/requirements.txt
  mkdir -p data/waypoint
  curl -L -o data/waypoint/best.pth \
      "https://drive.usercontent.google.com/download?id=1TsKqtdR1oir4UFIGhq15ffETz6r8-P2Q&export=download&confirm=t"   # SmartWay's waypoint checkpoint
  curl -L -o data/waypoint/gibson-2plus-resnet50.pth \
      https://zenodo.org/record/6634113/files/gibson-2plus-resnet50.pth                                        # DD-PPO depth encoder
  .conda/ac-wp/bin/python mip/tools/waypoint/predictor/server.py --port 9210   # leave it running
  curl http://127.0.0.1:9210/health                                            # {"status":"ok","name":"smartway_waypoint",…}
  ```

  The server checks both checkpoints against the sha256 of the files the paper used and loads the
  model before it opens the port. The network's RGB input is zeros, as it was in the paper's runs,
  so the candidates depend on depth alone (`predictor/engine.py`). The runner checks `/health`
  before episode 0 and stops when it does not answer.

## 7. Verify, in order

```
PY=.conda/coding-agents/bin/python
$PY runner.py std_mip_r2r_minimal harness=cc model=fable-5 +run.fake=true run.episodes=0   # a scripted agent: env server + data + tools, no tokens
$PY runner.py std_mip_r2r_minimal harness=cc model=fable-5 api=fake run.episodes=0        # the real Claude Code CLI on a scripted endpoint, no model
$PY runner.py std_mip_r2r_minimal harness=cc model=fable-5 run.episodes=0                 # one real episode on the subscription
$PY -m ui.server                                                                           # optional: http://<host>:8130, a read-only page over outputs/
```

Each step needs one more piece than the last: step 1 needs § 2, 3 and 5; step 2 a harness CLI
(cc runs the binary bundled in `claude-agent-sdk`, so it needs nothing from § 4; `harness=codex`
needs `codex` on PATH); step 3 the login. Steps 1 and 2 end with `distance_to_goal` 10.2291,
`ndtw` 0.1977 and 4 steps on every harness. The runs land in `outputs/fake/fake_mip_r2r_…`,
`outputs/claudecode/fakeapi_mip_r2r_…` and `outputs/claudecode/test_mip_r2r_…`, each with
`summary.json`, `stats.json`, `stats.html`, `config.yaml` (the resolved cell), `episode_0.jsonl`,
`live_0/` (every frame the model saw), `raw/` (the harness's own stream), `workdir_0/`,
`env_server.log` and `code/` (a copy of `mip/` and `runner.py`).

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: env.python='ac-es': no interpreter at …` | § 3 not done | § 3, or `env.python=/abs/path/to/python` |
| `No matching distribution` / `not a supported wheel on this platform` for habitat_sim | off the wheel's platform (x86_64 Linux, CPython 3.11) | build it (§ 3, second block) |
| `build.sh`: `need CMake >= 3.12 and < 4` | CMake 4 on PATH | `--cmake /path/to/cmake-3.x` |
| `import habitat_sim` → `CXXABI_1.3.15 not found` or `cannot retrieve OpenGL version` | a habitat build without the env RPATH (upstream's build.sh) | rebuild with the fork's `build.sh` |
| `libEGL.so.1` / `EGL context` errors | no graphics driver; in docker, the graphics capability is missing | install the driver; in docker set `NVIDIA_DRIVER_CAPABILITIES=graphics` |
| `500 Server Error … /configure`; `env_server.log`: `No Stage Attributes exists for requested scene` | the HM3D scene or its `scene_dataset_config.json` is not under `scenes/hm3d/` | § 5 |
| `500 Server Error … /configure`; `env_server.log`: `FileNotFoundError … hmeqa/…` | the HM-EQA corpus is not under `datasets/hmeqa/` | § 5 |
| `observe` returns `500 Server Error … /panorama`; `env_server.log`: `ValueError: Invalid mode for data type` | Pillow in `.conda/ac-es` is too new | `.conda/ac-es/bin/pip install pillow==10.4.0` (§ 3) |
| `ConnectionError … port=9210 … /health` | the waypoint predictor is not running at `run.wp_server` | § 6 |
| predictor `server.py`: `sha256 …, expected …` | a checkpoint download was cut short or is another file | download it again (§ 6) |
| `env_server.log`: `SlamDied: slam_server closed its stdout (the container died)` | docker not runnable by your user, or no `mip/orbslam3` image | `docker run --rm mip/orbslam3:latest true`; § 6 |
| cc harness log: `ANTHROPIC_API_KEY was set — removed` | by design: subscription auth | `CODING_AGENT_ALLOW_API_KEY=1` to bill via the API |
| `claude`: `/usr/bin/env: 'node': No such file or directory` | node is not on PATH in this shell | § 4; check `which claude` |
| `codex: command not found` from a script, fine in a terminal | nvm loads only in interactive shells | § 4 last paragraph |
