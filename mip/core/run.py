"""The run loop: one experiment run into its run directory, episode by episode (the body of
runner.py's launch; runner.py composes the config, builds the parts and hands them here).

The file is the loop too (region Runner): per run, let the harness prepare,
build the env pool (mip.core.envpool: the servers of ``run.servers``, the ones it
spawns, or whatever registers under ``run.registry`` while the run is on), let
the tool families check their run-level needs (``ToolSet.preflight``), snapshot
and start the callbacks (on_run_start), then the pool's workers pull episode
indices from a queue until it is empty or the run drains, each leasing a server
per episode — the pool health-checks and prepares a server for the whole run
(dataset / split) before its first lease.
Per episode: the env places it (``place(index)``: the placement payload — the
dataset's task statement and goal, the episode's identity, its goal count, the
action table), the tool set renders the briefing from its prompts group, the harness
runs one clean session through an EventBus, the env evaluates (``evaluate``), and the record is built from the metrics and the bridge Session's
report of how the episode ended (mip.core.toolset.read_report) — the callbacks
see every step (mip.core.callbacks.base for the hooks). A rate-limited session
is re-run in place (mip.core.retry), earlier attempts archived as
``*.attempt{k}*``. summary.json (mip.core.summary) is flushed after every episode;
``summary.provenance`` names the code, env, harness and callbacks
that produced the run, ``summary.writers`` maps every callback-owned key /
file to its class.

Graceful drain: while a run is live, ``touch <run.dir>/DRAIN`` (or
``kill -USR1 <pid>``) makes every worker finish its current episode, flush,
and exit WITHOUT starting a new one; the runner prints the un-run indices
as a ready-to-paste ``run.resume=true run.episodes=…`` for the next launch.

Region 1 holds what the loop reads and leaves behind — the episode lists, the attempt archive
and the flat run cfg projected from the four layers; region 2 is
``Runner``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from mip.core.bus import EventBus
from mip.core.callbacks import Callback, CallbackSet, StopRun
from mip.core.envclient import EnvClient
from mip.core.envpool import EnvPool
from mip.core.episode import EpisodeContext, ModelSpec, RunContext
from mip.harnesses.base import HarnessAdapter
from mip.core.panel import bridge_tool_schemas, json_safe
from mip.core.retry import DEFAULT_RETRY, RetryPolicy, is_rate_limited
from mip.core.summary import Summary, stub_record
from mip.core.toolset import read_report

# ---------------------------------------------------------------
# region 1. Records and the flat run cfg


# ── episode lists, attempts, caps ───────────────────────────────────────────


def format_episodes(indices: list[int]) -> str:
    """Inverse of parse_episodes: [7,8,9,14,44] -> '7-9,14,44'."""
    xs = sorted(set(indices))
    if not xs:
        return ""
    out: list[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def parse_episodes(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part:
            indices.append(int(part))
    return indices


def archive_attempt(run_dir: Path, index: int, attempt: int) -> None:
    """Before attempt k+1 re-runs an episode, move attempt k's files aside so
    every attempt is kept. The final attempt keeps the plain names, which is all
    run_stats / the monitor read."""
    suffix = f".attempt{attempt}"
    moves = [
        (run_dir / f"episode_{index}.jsonl", run_dir / f"episode_{index}{suffix}.jsonl"),
        (
            run_dir / "raw" / f"episode_{index}.jsonl",
            run_dir / "raw" / f"episode_{index}{suffix}.jsonl",
        ),
        (
            run_dir / "raw" / f"episode_{index}.stderr.log",
            run_dir / "raw" / f"episode_{index}{suffix}.stderr.log",
        ),
        (  # a harness process's episode spec (mini)
            run_dir / "raw" / f"episode_{index}.spec.json",
            run_dir / "raw" / f"episode_{index}{suffix}.spec.json",
        ),
        (  # mini's own trajectory
            run_dir / "raw" / f"episode_{index}.traj.json",
            run_dir / "raw" / f"episode_{index}{suffix}.traj.json",
        ),
        (
            run_dir / "raw" / f"context_manifest_{index}.json",
            run_dir / "raw" / f"context_manifest_{index}{suffix}.json",
        ),
        (run_dir / f"live_{index}", run_dir / f"live_{index}{suffix}"),
    ]
    for src, dst in moves:
        if src.exists():
            src.rename(dst)


# ── the flat run cfg ───────────────────────────────────────────────────────

# env keys the loop reads by name: which cfg keys the env's configure() takes (the task's own
# keys — rgb_resolution among them — are already in cfg; the tool set adds its families' own,
# ToolSet.env_fields)
ENV_CFG_KEYS = ("env_fields",)


def model_extra(model: dict[str, Any], api: dict[str, Any] | None = None) -> dict[str, Any]:
    """The API routing params beside the model id (api_base / api_gateway /
    gateway_route …) and the model's sampling ``params`` (temperature /
    max_tokens …, consumed by harnesses that expose them — mini; cc and
    codex refuse them): everything in ``agent.model`` but ``id``, then the
    experiment's api/ block (``agent.api``, picked with ``api=<name>``) on
    top, unset (null) values left out. Effort is NOT one of them — it is
    the top-level ``effort`` key, an explicit parameter of every harness
    adapter."""
    if "effort" in model:
        sys.exit(
            "[std] agent.model.effort is not a knob — set the top-level `effort` key (effort=max …)"
        )
    extra = {k: v for k, v in model.items() if k != "id" and v is not None}
    extra.update({k: v for k, v in (api or {}).items() if v is not None})
    return extra


def run_cfg(app: dict[str, Any], tool_set: Any) -> dict[str, Any]:
    """The flat run cfg from the four layers: the task's data + budgets,
    the env's stack facts, the run's fuse and the
    model's params — the dict the loop (``cfg["dataset"]``), the episode
    context and the recorders read. No defaults of its own: every key is
    present exactly when the config declares it. The tool set's switches stay on it."""
    task, env, run = app["task"], app["env"], app["run"]
    cfg: dict[str, Any] = {k: v for k, v in task.items() if k != "family"}
    for key in ENV_CFG_KEYS:
        if key in env:
            cfg[key] = env[key]
    cfg["wp_server"] = run.get("wp_server") if tool_set.needs_wp_server else None
    if run.get("budget_usd") is not None:  # run-level fuse, see callbacks.CostLogger
        cfg["budget_usd"] = float(run["budget_usd"])
    cfg["extra"] = model_extra(app["agent"]["model"], app["agent"].get("api"))
    return cfg


# endregion
# ---------------------------------------------------------------
# region 2. Runner


class Runner:
    def __init__(
        self,
        adapter: HarnessAdapter,
        tool_set: Any,
        app: dict[str, Any],
        callbacks: list[Callback],
        retry: RetryPolicy = DEFAULT_RETRY,
        env_client: Callable[..., Any] = EnvClient,
    ) -> None:
        self.adapter = adapter
        self.tool_set = tool_set
        self.app = app
        self.name: str = app["run"]["name"]
        self.env_name: str = str(app["env"]["_target_"]).rpartition(".")[
            2
        ]  # the env object's class
        # how the loop reaches an env server (url, timeout_s=…) -> EnvClient's surface; a
        # caller without a server can hand it a stand-in env
        self.env_client = env_client
        self.task_key: str = app["task_key"]
        self.model = ModelSpec(
            app["agent"]["model"]["id"],
            effort=str(app.get("effort") or "default"),
            extra=model_extra(app["agent"]["model"], app["agent"].get("api")),
        )
        self.cfg = run_cfg(app, tool_set)
        self._callbacks = list(callbacks)
        self.retry = retry
        # run.client_timeout_s: one env verb over HTTP (place / step / evaluate …)
        self.client_timeout_s = float(app["run"].get("client_timeout_s") or 600.0)
        self.run_ctx: RunContext | None = None  # the run in progress

    # ── one episode ──

    async def run_episode(self, cfg: dict[str, Any], url: str, index: int) -> dict[str, Any]:
        run = self.run_ctx
        assert run is not None
        env = self.env_client(url, timeout_s=self.client_timeout_s)

        # the placement: what the env's place(index) returned — the dataset's task
        # statement and goal, the episode's identity, its goal count, the action table
        placement = await env.call("place", index=index)
        if "error" in placement:
            raise RuntimeError(f"{url} place({index}): {placement['error']}")
        if not placement.get("episode_id"):
            raise RuntimeError(
                f"{self.env_name} place({index}) returned no episode — placement failed?"
            )
        prompts = self.tool_set.episode_prompts(placement, cfg["step_budget"])
        statement = prompts.statement
        if not statement:
            raise RuntimeError(
                f"{self.env_name} place({index}) returned an empty task statement "
                f"(goal {placement.get('goal')!r})"
            )
        task_word = str(placement.get("task_word") or "nav")
        briefing = prompts.briefing

        workdir = run.run_dir / f"workdir_{index}"
        workdir.mkdir(parents=True, exist_ok=True)
        raw_dir = run.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        ctx = EpisodeContext(
            index=index,
            instruction=statement,
            briefing=briefing,
            first_prompt=prompts.first_prompt,
            server_url=url,
            model=self.model.model_id,
            effort=self.model.effort,
            max_turns=cfg["max_turns"],
            step_budget=cfg["step_budget"],
            episode_timeout=cfg["episode_timeout"],
            workdir=workdir,
            live_dir=run.run_dir / f"live_{index}",
            raw_dir=raw_dir,
            benchmark=self.task_key,
            tool_set=self.tool_set,
            max_budget_usd=cfg.get("max_budget_usd"),
            wp_server_url=cfg.get("wp_server") or "",
            extra=dict(cfg.get("extra") or {}),
            run_dir=run.run_dir,
            task_word=task_word,
            # the line's action table: the task's own declaration (4 on the R2R / RxR protocol)
            # over what the env placed (the STANDARD 6)
            n_actions=int(cfg["n_actions"]) if cfg.get("n_actions") else placement.get("n_actions"),
        )

        callbacks = run.callbacks
        bus = EventBus(ctx, callbacks)
        ctx.bus = bus
        record: dict[str, Any] | None = None
        callbacks.fire("on_episode_start", ctx)
        try:
            bus.emit(
                "episode_meta",
                {
                    "index": index,
                    "episode_id": placement.get("episode_id"),
                    "scene_id": placement.get("scene_id"),
                    "instruction": statement,
                },
            )
            bus.emit(
                "session_inputs",
                {
                    "cell": self.name,
                    "harness": self.adapter.name,
                    "model": self.model.model_id,
                    "system_prompt": briefing,
                    "first_prompt": ctx.first_prompt,
                    # The recorded surface is the one the agent gets: same bridge,
                    # same environment the session hands it.
                    "tool_schemas": await bridge_tool_schemas(ctx.bridge_path, ctx.bridge_env()),
                    **json_safe(self.adapter.describe(ctx)),
                },
            )

            outcome = await self.adapter.run(ctx, bus)

            bus.emit(
                "result",
                {
                    "result": json_safe(
                        {
                            "usage": outcome.usage,
                            "cost_usd": outcome.cost_usd,
                            "turns": outcome.turns,
                            "error": outcome.error,
                            **outcome.extra,
                        }
                    )
                },
            )

            # Evaluate while the trajectory is still open so the metrics land
            # in the log itself. Driver-side; the agent never sees it.
            metrics = await self._evaluate(env, bus)
            bus.emit("episode_metrics", {"metrics": metrics})
            wall = bus.elapsed

            # how the episode ended: the bridge Session's report, else the last step result
            last = read_report(ctx.live_dir) or bus.last_step_result or {}
            record = {
                "index": index,
                "episode_id": placement.get("episode_id"),
                "scene_id": placement.get("scene_id"),
                "instruction": statement,
                "metrics": metrics,
                "agent": {
                    "tool_calls": bus.tool_calls,
                    "env_steps": last.get("steps_taken_total", 0),
                    "end_reason": last.get("end_reason"),
                    # deliberate termination: STOP on the nav lines, answer() on
                    # hmeqa — same "the agent chose to end" semantics either way
                    "called_stop": last.get("end_reason")
                    in ("stop_called", "answer_submitted"),
                    "num_turns": outcome.turns,
                    "usage": outcome.usage,
                    "total_cost_usd": outcome.cost_usd,
                    **json_safe(outcome.extra),
                    # the agent's answer, when it gave one (answer()), and the task word placed
                    **({"answer": last["answer"]} if last.get("answer") else {}),
                    "task_word": task_word,
                },
                "wall_sec": round(wall, 1),
            }
            if outcome.error:
                record["error"] = outcome.error
        finally:
            try:
                callbacks.fire("on_episode_end", ctx, record)
            except StopRun as stop:
                self._stop(stop)
        return record

    async def _evaluate(self, env: EnvClient, bus: EventBus) -> dict[str, Any]:
        """The line's own ruler, driver-triggered; the agent never sees it. The env
        holds the agent's answer itself (its submit_answer verb).
        A failed evaluate is a driver_error and an empty metrics dict."""
        try:
            metrics = await env.call("evaluate")
            if "error" in metrics:
                raise RuntimeError(metrics["error"])
            return metrics
        except Exception as exc:
            bus.emit("driver_error", {"error": f"evaluate failed: {exc!r}"})
            return {}

    def _stop(self, stop: StopRun) -> None:
        run = self.run_ctx
        assert run is not None
        if run.stopped_by is None:
            run.stopped_by = str(stop)
            print(f"[std] STOP requested by a callback: {stop} — draining")
        run.drain.set()

    # ── the run ──

    async def run(self, run_dir: Path) -> dict[str, Any]:
        """Run (or resume) one experiment into ``run_dir``. Existing episode
        records in the run dir's summary are kept; requested indices
        (``run.episodes``) are re-run and replace their records."""
        # May raise (bad auth, unpinnable serving context) — that is the point.
        self.adapter.prepare(self.model)
        # the env pool (mip.core.envpool: run.registry, run.servers, or spawned from the env block),
        # each server prepared for this run before its first lease
        pool = EnvPool.for_run(self.app["run"], self.app["env"], run_dir, prepare=self._configure)
        try:
            return await self._run(run_dir, pool)
        finally:
            pool.close()

    async def _configure(self, url: str) -> None:
        """A server's preparation for this run (the pool runs it before the server's first
        lease): the health guard — it must serve this run's env; a mismatched server would
        place the WRONG episodes, so fail loud rather than degrade — and the placement for the
        whole run: one ``configure`` with the task's line and split plus the run-level fields
        — the cfg keys ``env_fields`` names and the tool families' own (``ToolSet.env_fields``:
        slam's ``instruments``), pushed BEFORE any placement so the first seat already builds
        the right stack. Nothing per benchmark."""
        cfg = self.cfg
        env = self.env_client(url, timeout_s=self.client_timeout_s)
        health = await asyncio.to_thread(env.health)
        print(f"[std] {url} healthy: {health['name']}")
        if health["name"] != self.env_name:
            raise RuntimeError(
                f"{url} serves {health['name']!r} but run {self.name} needs {self.env_name!r}"
            )
        extra = {name: cfg[name] for name in (cfg.get("env_fields") or ()) if name in cfg}
        extra.update(self.tool_set.env_fields())
        out = await env.call("configure", line=cfg["dataset"], split=cfg["split"], **extra)
        if "error" in out:
            raise RuntimeError(f"{url} configure({cfg['dataset']}, {cfg['split']}): {out['error']}")

    async def _run(self, run_dir: Path, pool: EnvPool) -> dict[str, Any]:
        adapter, cfg, app = self.adapter, self.cfg, self.app
        run_name, env_name = self.name, self.env_name
        episodes_spec: str = str(app["run"]["episodes"])
        workers = pool.workers
        server_timeout_s = float(app["run"].get("server_timeout_s") or 600.0)

        # at least one env server before anything starts; with a registry, up to run.workers
        # of them may still be on their way and join later
        known = await pool.wait_for(workers, timeout_s=server_timeout_s)

        # each tool family's run-level check (the waypoint predictor, the SLAM hook) — refuse
        # before episode 0
        self.tool_set.preflight(
            wp_server=app["run"].get("wp_server"),
            dataset=str(cfg.get("dataset") or ""),
            env=app["env"],
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        # the ONE full record of what this launch runs on: the resolved
        # four-layer config, plus the flat run cfg the loop derived from it
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {**json_safe(app), "run_cfg": json_safe(cfg)}, sort_keys=False, allow_unicode=True
            )
        )

        # resume: keep prior records for indices not being re-run
        prior = Summary.prior(run_dir / "summary.json")
        run = RunContext(
            name=run_name,
            app=app,
            cfg=cfg,
            run_dir=run_dir,
            servers=pool.urls,
            adapter=adapter,
            tool_set=self.tool_set,
            env_name=env_name,
            episodes_spec=episodes_spec,
            prior_episodes=prior,
            episodes=dict(prior),
        )
        callbacks = CallbackSet(self._callbacks, run)
        run.callbacks = callbacks
        run.provenance.update(
            {
                "task": self.task_key,
                "env_key": app["env_key"],
                "arm": self.tool_set.name,  # provenance.arm: the interface (tool set) name
                "env": env_name,
                "harness": adapter.name,
                "callbacks": callbacks.names(),
            }
        )
        self.run_ctx = run
        summary = Summary(
            run,
            lambda: {
                "run_name": run_name,
                "cell": run_name,
                "harness": adapter.name,
                "harness_inherent": adapter.inherent,
                "config": {
                    **{k: v for k, v in cfg.items() if k != "extra"},
                    "model": self.model.model_id,
                    "effort": self.model.effort,
                    "bare": self.tool_set.bare,
                    "extra": json_safe(cfg["extra"]),
                },
                "servers": pool.urls,
            },
        )
        callbacks.fire("on_run_start", run)

        # Graceful drain:
        # `touch <run_dir>/DRAIN` (or SIGUSR1) asks
        # every worker to finish its current episode, flush, and exit WITHOUT
        # pulling a new one — in-flight work is never cut, un-pulled indices stay
        # pending for the next resume. A callback's StopRun takes the same path.
        drain = run.drain
        drain_sentinel = run_dir / "DRAIN"
        if drain_sentinel.exists():
            drain_sentinel.unlink()  # clear a stale sentinel from a prior run
        with contextlib.suppress(
            NotImplementedError, RuntimeError
        ):  # unavailable on this platform / thread
            asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, drain.set)

        indices = parse_episodes(episodes_spec)
        print(
            f"[std] run={run_name} model={self.model.model_id} eps={indices[0]}-{indices[-1]} "
            f"workers={workers} env servers={known} -> {run_dir}"
        )

        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in indices:
            queue.put_nowait(index)
        episodes = run.episodes

        async def run_index(index: int, url: str) -> dict[str, Any]:
            """One episode on a leased server, with the rate-limit retry (without it a
            throttled episode lands as a contaminated partial record instead of re-running):
            back off OUTSIDE the timed scope and re-run the episode fresh, so the wait never
            eats the episode's wall-clock budget."""
            episode: dict[str, Any] = {}
            for attempt in range(1, self.retry.max_attempts + 1):
                if attempt > 1:
                    archive_attempt(run_dir, index, attempt - 1)
                try:
                    episode = await asyncio.wait_for(
                        self.run_episode(cfg, url, index),
                        timeout=cfg["episode_timeout"] + 600,  # backstop over in-session caps
                    )
                except asyncio.TimeoutError:
                    print(f"[std] episode {index} TIMED OUT (backstop)")
                    episode = stub_record(index, "timeout")
                except Exception as exc:
                    tag = "rate_limited" if is_rate_limited(repr(exc)) else repr(exc)
                    print(f"[std] episode {index} FAILED: {exc!r}")
                    episode = stub_record(index, tag)
                if episode.get("error") != "rate_limited" or attempt == self.retry.max_attempts:
                    break
                backoff = self.retry.backoff(attempt)
                print(
                    f"[std] episode {index} RATE-LIMITED (attempt {attempt}/"
                    f"{self.retry.max_attempts}) — backing off {backoff}s "
                    f"(episode countdown paused) then retrying"
                )
                await asyncio.sleep(backoff)  # OUTSIDE wait_for: excluded from episode_timeout
            return episode

        async def worker(position: int) -> None:
            await asyncio.sleep(position * 2)  # stagger cold-server scene loads
            while True:
                if drain.is_set() or drain_sentinel.exists():
                    print(f"[std] worker {position} draining — no new episode pulled")
                    return
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                url = await pool.lease()
                try:
                    print(f"[std] episode {index} starting on {url}")
                    episode = await run_index(index, url)
                finally:
                    pool.release(url)
                episodes[index] = episode
                await summary.flush()
                m = episode.get("metrics") or {}
                # steps come from the env's own metrics, else the agent-side counter
                # (display only)
                _dtg = m.get("distance_to_goal")
                _steps = m.get("steps_taken") or episode["agent"].get("env_steps")
                print(
                    f"[std] episode {index} done: success={m.get('success')} "
                    f"spl={m.get('spl')} "
                    f"dtg={round(_dtg, 2) if isinstance(_dtg, (int, float)) else None} "
                    f"steps={_steps}"
                )

        try:
            await asyncio.gather(*(worker(i) for i in range(workers)))
        finally:
            if pool.dropped:
                run.run_stats["env_servers_dropped"] = list(pool.dropped)
            built = summary.build()
            with contextlib.suppress(StopRun):  # the run is over either way
                callbacks.fire("on_run_end", run, built)
            if run.stopped_by:
                run.run_stats["stopped_by"] = run.stopped_by
            if run.callback_errors:
                run.run_stats["callback_errors"] = dict(run.callback_errors)
            finalize = getattr(adapter, "finalize", None)
            if callable(finalize):
                try:
                    run.run_stats.update(finalize(run_dir) or {})
                except Exception as exc:
                    run.run_stats["finalize_error"] = repr(exc)
            # on_run_end may have added run-level keys; rebuild so run_stats /
            # aggregate / episodes reflect everything before the final write
            summary.absorb(built)
            await summary.flush()

        if drain.is_set() or drain_sentinel.exists():
            left: list[int] = []
            while not queue.empty():
                left.append(queue.get_nowait())
            print(
                f"[std] DRAINED — in-flight episodes finished; "
                f"{len(left)} un-run, resume with --resume --episodes {format_episodes(left)}"
                if left
                else "[std] DRAINED — queue already empty"
            )
        if drain_sentinel.exists():
            drain_sentinel.unlink()

        final = summary.scored()
        print(f"[std] run complete -> {run.summary_path}")
        if run.run_stats:
            print(f"[std] run stats: {json.dumps(run.run_stats)}")
        print(json.dumps(final, indent=2))
        callbacks.fire("on_run_written", run, run.summary_path)
        return final


# endregion
