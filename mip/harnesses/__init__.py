"""Harnesses — what runs an episode: the contract every harness implements, and one adapter per
agent stack.

The contract:

    base.py         HarnessAdapter (prepare / describe / run / finalize), the frozen Harness record
                    of the stack an adapter wraps, the one effort rule, and the shared helpers
    process.py      the session loop of a harness that runs as its own process (ProcessAdapter;
                    ProtocolAdapter for the ones MIP launches itself)
    protocol.py     what crosses that process boundary: the episode spec in, event lines out
    fake.py         the token-free scripted harness for wiring checks (+run.fake=true)

The adapters, each running ONE clean session per episode and emitting through the shared EventBus:

    claude_sdk.py   Claude Code, through the Claude Agent SDK
    codex_cli.py    Codex, through ``codex exec --json``
    mini_swe.py     mini-swe-agent, its loop over the experiment's bridge (mini/main.py, the process)

Everything else (placement, prompts, evaluation, artifacts, summary) lives in runner.py,
mip.core.toolset, the env's own verbs and mip.core.callbacks. An adapter is instantiated by
runner.py from the ``agent.harness`` block of the experiment config (``_target_`` + the harness's
constants as keyword arguments; mip/configs/harness/); an unknown key is a TypeError from the
constructor.
"""

from __future__ import annotations
