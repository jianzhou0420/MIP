"""mip — the Minimal-Interface Probe.

    core/       the framework: the run loop, the tool set and its MCP bridge, the env server, its
                client and pool, the API layer and the recorders
    harnesses/  the harness contract and the three adapters (Claude Code SDK, Codex CLI, mini-SWE-agent)
    tools/      what the agent can call: nav, waypoint, slam
    prompts/    the briefings, one per interface and benchmark
    configs/    the experiment cells and the harness / models / api groups
    envs/       the env objects the env server serves (EmbodiedScoreEnv)
    reporting/  per-run statistics

The entry point is runner.py at the repo root."""
