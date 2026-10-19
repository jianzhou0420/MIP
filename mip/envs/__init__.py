"""Envs — the robot and its environment, one package per env, each served by mip.core.envserver.

An env object is the ``env`` layer of an experiment file (``_target_`` + its kwargs), served in
the interpreter the file names (``env.python``) and reached over HTTP by the runner (place,
configure, evaluate) and by the agent's tools (observe, step …): every public method is a verb.
It provides raw capabilities only — the sensors of its body, the actions it can execute, the
episodes and the benchmark's own metrics; what the agent is shown is the tools' business.

    embodiedscore/  EmbodiedScore-envs, the simulated lines (VLN-CE, ObjectNav, EQA)
"""
