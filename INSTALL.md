# Install

Two ways to get the simulator; everything else is the same.

| | A. Install from wheels (default) | B. Build the simulator from source |
|---|---|---|
| for | x86_64 Linux, CPython 3.10 – 3.13 | anything else: aarch64 / Jetson, Python 3.9, a bullet-enabled build |
| needs | a graphics driver (NVIDIA; mesa works, slower), node 22 for the two CLIs | the same, plus CMake ≥ 3.12 and < 4, ninja, GCC ≥ 9, EGL headers, ~10 min |
| where | §1 – §4 below | § B, then §2 – §4 |

## A. Install from wheels

### 1. Python

```
python3 -m venv envs/mip && . envs/mip/bin/activate     # or: conda create -p envs/mip python=3.11
pip install -r requirements.txt                          # simulator wheel + env package + runner, ~2 min
python -m pytest tests -q                                # no GPU, no data, no keys
```

### 2. The coding agents

```
npm install -g @anthropic-ai/claude-code @openai/codex   # node 22: https://github.com/nvm-sh/nvm
claude          # log in once (subscription); the runner strips ANTHROPIC_API_KEY so sessions never bill via API
codex login     # once
```

mini-SWE-agent needs the provider key in the shell: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`DASHSCOPE_API_KEY` (qwen rows) or `OLLAMA_URL` (local rows). Rows: `exp_workspace/bareES/configs/models/models.yaml`.

### 3. Data

MP3D scenes are licensed (Matterport terms) and not redistributed; R2R-CE / RxR-CE episodes are
public. The evaluation splits (`rand100`, `mip100`) ship in `splits/`. Link what you have:

```
mkdir -p data/embodiedscore/{datasets,scenes}
ln -s /path/to/vlnce   data/embodiedscore/datasets/vlnce    # R2R_VLNCE_v1-3_preprocessed/, RxR_VLNCE_v0/
ln -s /path/to/mp3d    data/embodiedscore/scenes/mp3d       # <scan>/<scan>.glb + .navmesh, 90 scans
ln -sfn "$PWD/splits/r2r/rand100" data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed/rand100
ln -sfn "$PWD/splits/rxr/rand100" data/embodiedscore/datasets/vlnce/RxR_VLNCE_v0/rand100
```

Each experiment yaml lists its files under `task.require`; VLNVerse and HM-EQA need their own
corpora (`datasets/vlnverse/`, `datasets/hmeqa/` + `scenes/hm3d/`). Runs land in `outputs/`.

### 4. Verify, in order

```
python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0   # scripted agent: simulator + data + tools, no tokens
python runner.py std_r2r_es_bareES harness=cc model=fable-5 api=fake run.episodes=0        # the real Claude Code CLI on a scripted endpoint, no model
python runner.py std_r2r_es_bareES harness=cc model=fable-5 run.episodes=0                 # one real episode
```

## B. Build the simulator from source

The wheels are habitat-sim 0.3.3 built from the
[EmbodiedScore-habitat](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat) fork, headless,
without bullet, with the magnum bindings inside. Off the wheel matrix, build the same thing into your
env with the fork's `build.sh` (its `BUILD.md` has the system packages and the compiler matrix):

```
python3 -m venv envs/mip && . envs/mip/bin/activate
git clone --recursive https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat.git   # ~2.3 GB with submodules
pip install -r EmbodiedScore-habitat/requirements.txt                                   # build.sh wants the runtime deps in place
(cd EmbodiedScore-habitat && ./build.sh --python "$(command -v python)" --verify)        # ~10 min; --verify renders one EGL frame
grep -v '^habitat_sim @' requirements.txt | pip install -r /dev/stdin                    # everything but the wheel lines
```

Then continue at §2. To make your own wheels instead (another architecture, a bullet build), the
fork's `pyproject.toml` carries the `cibuildwheel` recipe: `cibuildwheel --platform linux .` with
docker present, or push a `v*` tag and its GitHub workflow builds and attaches them.

## Troubleshooting

| symptom | fix |
|---|---|
| `libEGL.so.1` / `EGL context` errors | install the graphics driver; in Docker add `NVIDIA_DRIVER_CAPABILITIES=graphics` |
| `No matching distribution found for habitat_sim` / `not a supported wheel on this platform` | off the wheel matrix (CPython 3.10 – 3.13, x86_64 Linux): go to § B |
| `claude`: `'node': No such file or directory` | node is not on PATH in this shell (nvm loads only in interactive shells) |
