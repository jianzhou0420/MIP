# MIP — Minimal-Interface Probe

Code for **Embodied Agents Take Control: Minimal-Interface Zero-Shot Agents Rival Industrial-Scale
Policies in Vision-and-Language Navigation** (arXiv:2607.26148).

A general-purpose coding agent (Claude Code, Codex CLI, or mini-SWE-agent) is put in front of a
simulator through two tools, `observe()` and `step()`, with no map, memory, waypoint predictor,
search or training, and scored on R2R-CE, RxR-CE, VLNVerse and HM-EQA exactly as the trained
systems are. The environments, episodes and metrics come from
[EmbodiedScore-envs](https://github.com/Embodied-Agent-Squad/EmbodiedScore-envs); the simulator is
[EmbodiedScore-habitat](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat) (habitat-sim
0.3.3, shipped as a wheel).

Install: [INSTALL.md](INSTALL.md) — one interpreter, `pip install -r requirements.txt`, link the data,
three commands to verify.

## Run

```
python runner.py std_r2r_es_bareES harness=cc model=fable-5                   # one board cell: R2R-CE rand100, Claude Code, fable-5
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.6 effort=low     # Codex CLI
python runner.py std_r2r_es_bareES harness=mini model=qwen3.6-plus            # mini-SWE-agent
python runner.py std_r2r_es_bareES harness=cc model=fable-5 run.episodes=0-9  # a probe: ten episodes
python -m ui.server                                                           # a read-only page over outputs/
```

One yaml is one experiment (`exp_workspace/bareES/configs/`): task, environment, agent and run are
written out in full; only the seat is chosen on the command line — `harness=` (cc · codex · mini),
`model=` (a row of `configs/models/models.yaml`), `effort=` (the vendor's word; omitted = default).
The run name `std_<task>_es_<harness>_<model>_<effort>_bareES` is the cell's identity.

**The paper's cells**, table by table, are the lines of [`scripts/mip_paper_cells.sh`](scripts/mip_paper_cells.sh)
(a reference to copy from, not a program). Not in this repository: the waypoint-interface and hybrid
arms (Tables 4–6, waypoint rows), the physical-robot probes (Appendix D) and the human row.

## Layout

```
runner.py                 the entry point and the run loop
core/                     the framework: arm · episode · shapes · harnesses/ (the three coding-agent loops
                          + the scripted FakeHarness) · callbacks/ (logs, cost, code snapshot, stats) · api/ (the
                          scripted endpoint for api=fake) · envserver / envclient (the simulator in its own process)
exp_workspace/bareES/     the minimal-interface arm: prompts.py (the briefing) · mcp/env.py (the world as a verb
                          object) · mcp/bridge.py (the verbs as MCP tools) · configs/ (R2R-CE, RxR-CE, VLNVerse, HM-EQA)
splits/                   the evaluation splits (r2r/rand100, rxr/rand100, hmeqa/mip100) and their provenance
scripts/                  mip_paper_cells.sh
reporting/ · ui/          per-run statistics and the results page
tests/                    the framework's tests (no GPU, no data, no keys)
```

Every run writes `summary.json`, `episode_<i>.jsonl`, `live_<i>/` frames, `env_server.log` and
`stats.html` under `outputs/<harness>/<run name>`; the agent never sees a score.
