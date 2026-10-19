"""EnvPool — the env servers a run may lease, one episode at a time.

The run loop (mip.core.run.Runner._run) leases a server for every episode and hands it back after;
where the servers come from is the pool's business: the addresses the run was given
(``run.servers``), the ones it spawns itself (``mip.core.envserver`` children, stopped by ``close``)
and the registrations under a directory (``run.registry``), where a server started elsewhere — a
cluster job, another shell — writes one ``Registration`` when it is up and removes it when it
exits, so servers join and leave while the run is on. A server that stops answering ``/health``
is dropped at its next lease. A leased server is prepared: the pool runs the run's ``prepare``
on each server once, before its first lease.

    pool = EnvPool.for_run(app["run"], app["env"], run_dir, prepare=configure)
    await pool.wait_for(pool.workers, timeout_s=600)  # at least one server, or NoEnvServer
    url = await pool.lease()                    # a free, healthy, prepared server; waits while all are busy
    ...
    pool.release(url)
    pool.close()                                # the spawned servers stop

mip.core.envserver writes the registration (``Registration.write``); this module reads it.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class NoEnvServer(RuntimeError):
    """The pool holds no live env server and none arrived in time."""


@dataclass(frozen=True)
class Registration:
    """What an env server writes under the registry directory when it is up. The file is
    ``<registry>/<host>:<port>.json``, removed when the server exits; one left by a killed server
    is dropped by the pool when the address stops answering."""

    address: str  #: host:port as another node reaches it (the hostname, not a loopback address)
    env: str  #: the env object's class name, what /health reports as ``name``
    pid: int
    started: str  #: ISO-8601 UTC; a restarted server on the same port has a new one

    @classmethod
    def of(cls, env_name: str, host: str, port: int) -> Registration:
        reachable = socket.gethostname() if host in ("0.0.0.0", "") else host
        return cls(
            f"{reachable}:{port}",
            env_name,
            os.getpid(),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    @property
    def url(self) -> str:
        return f"http://{self.address}"

    def write(self, registry: Path) -> Path:
        """The registration file, written whole (write then rename: a reader never sees a part)."""
        registry.mkdir(parents=True, exist_ok=True)
        path = registry / f"{self.address}.json"
        part = path.with_suffix(".part")
        part.write_text(json.dumps(asdict(self)) + "\n")
        part.replace(path)
        return path

    @classmethod
    def read(cls, path: Path) -> Registration:
        return cls(**json.loads(path.read_text()))


class EnvPool:
    """The servers a run knows, by url; ``lease`` marks one busy, ``release`` frees it."""

    def __init__(
        self,
        *,
        registry: Path | None = None,
        servers: list[str] = (),
        poll_s: float = 1.0,
        prepare: Callable[[str], Awaitable[None]] | None = None,
        workers: int = 1,
        spawned: list[Any] = (),
    ) -> None:
        self.registry = registry
        self.poll_s = poll_s  #: between looks for a freed or a newly registered server
        self.workers = workers  #: how many episodes the run keeps in flight
        self._prepare = prepare  #: run on each server once, before its first lease
        self._prepared: set[str] = set()
        self._spawned = list(spawned)  #: mip.core.envserver handles this pool stops on close
        self._started: dict[str, str | None] = {url: None for url in servers}  # url -> started
        self._busy: set[str] = set()
        self._gone: dict[str, str | None] = {}  #: dropped url -> the registration it had
        self.dropped: list[str] = []  #: every url dropped so far, for the run's record

    @classmethod
    def for_run(
        cls,
        run: dict[str, Any],
        env: dict[str, Any],
        log_dir: Path,
        prepare: Callable[[str], Awaitable[None]] | None = None,
    ) -> EnvPool:
        """The pool of one run block: the servers registering under ``run.registry``
        (started elsewhere, joining as they register), the ones ``run.servers`` names, or
        ``run.workers`` spawned here from the env block (own interpreter, ``env.python``;
        ports ``env.port``, ``env.port+1``, …; logs in ``log_dir``), waited for up to
        ``run.server_timeout_s``. ``workers`` is the episodes in flight: ``run.workers``, or
        one per named server."""
        workers = int(run.get("workers") or 1)
        servers = list(run.get("servers") or [])
        if run.get("registry"):
            if servers:
                raise ValueError(
                    "run.registry and run.servers both name the env servers — give one"
                )
            registry = Path(run["registry"])
            print(f"[std] env pool: the servers registering under {registry}")
            return cls(registry=registry, prepare=prepare, workers=workers)
        if servers:
            if workers != 1 and workers != len(servers):
                raise ValueError(
                    f"run.workers={workers} but run.servers lists {len(servers)} — give one or the other"
                )
            return cls(servers=servers, prepare=prepare, workers=len(servers))
        from mip.core import envserver  # here: the env server's interpreter imports this module too

        base = int(env.get("port") or 9200)
        # one server: env_server.log; several: one log per port
        handles = [
            envserver.launch(
                env,
                base + i,
                log_dir / (f"env_server_{base + i}.log" if workers > 1 else "env_server.log"),
            )
            for i in range(workers)
        ]
        try:
            envserver.wait_healthy(handles, timeout_s=float(run.get("server_timeout_s") or 600.0))
        except BaseException:
            for handle in handles:
                handle.stop()
            raise
        env_name = str(env["_target_"]).rpartition(".")[2]
        print(
            f"[std] {workers} env server(s) {env_name} spawned on {base}-{base + workers - 1} (logs: {log_dir})"
        )
        return cls(
            servers=[h.url for h in handles], prepare=prepare, workers=workers, spawned=handles
        )

    def close(self) -> None:
        """Stop the servers this pool spawned; the others are not the run's to stop."""
        for handle in self._spawned:
            handle.stop()
        self._spawned.clear()

    @property
    def urls(self) -> list[str]:
        return sorted(self._started)

    @property
    def size(self) -> int:
        return len(self._started)

    def refresh(self) -> list[str]:
        """Pick up the registrations written since the last look; the urls that joined. A dropped
        server comes back only with a new registration (a restart on the same port)."""
        if self.registry is None:
            return []
        joined: list[str] = []
        for path in sorted(self.registry.glob("*.json")):
            try:
                reg = Registration.read(path)
            except (OSError, ValueError, TypeError):
                continue  # being written, or gone between the glob and the read
            if reg.url in self._started or self._gone.get(reg.url) == reg.started:
                continue
            self._started[reg.url] = reg.started
            joined.append(reg.url)
        return joined

    async def wait_for(self, count: int, timeout_s: float) -> int:
        """Look at the registry until ``count`` servers are known or ``timeout_s`` passed; the
        number known. NoEnvServer when there is none at all — a run cannot start without one."""
        deadline = time.monotonic() + timeout_s
        self.refresh()
        while self.size < count and time.monotonic() < deadline:
            await asyncio.sleep(min(5.0, max(self.poll_s, timeout_s / 100)))
            self.refresh()
        if not self._started:
            where = f" under {self.registry}" if self.registry is not None else ""
            raise NoEnvServer(f"no env server after {timeout_s:.0f} s{where}")
        return self.size

    async def lease(self) -> str:
        """A free server that answers /health, marked busy and prepared. Waits while every known
        server is busy, looking at the registry meanwhile; NoEnvServer once the pool has run dry.
        A failed ``prepare`` frees the server and raises."""
        while True:
            self.refresh()
            for url in self.urls:
                if url in self._busy:
                    continue
                if await asyncio.to_thread(_healthy, url):
                    self._busy.add(url)
                    if self._prepare is not None and url not in self._prepared:
                        try:
                            await self._prepare(url)
                        except BaseException:
                            self.release(url)
                            raise
                        self._prepared.add(url)
                    return url
                self._drop(url)
            if not self._started:
                raise NoEnvServer("every env server of the run is gone")
            await asyncio.sleep(self.poll_s)

    def release(self, url: str) -> None:
        self._busy.discard(url)

    def _drop(self, url: str) -> None:
        self._gone[url] = self._started.pop(url, None)
        self._busy.discard(url)
        self.dropped.append(url)


def _healthy(url: str) -> bool:
    import requests  # the env server's interpreter imports this module too and has no requests

    try:
        return requests.get(f"{url}/health", timeout=5).ok
    except requests.RequestException:
        return False
