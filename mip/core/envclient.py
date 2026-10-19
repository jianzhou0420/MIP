"""envclient — the one HTTP client for an env server (mip.core.envserver).

    env = EnvClient("http://127.0.0.1:9200")
    ep = await env.call("place", index=3)          # runner side: never blocks the loop
    rgb = env.call_sync("observe")["rgb"]           # bridge side (mip.core.toolset.Session.call)

A verb is ``POST /<name>`` with the kwargs as JSON; the server's reply is
returned as-is (a dict with ``error`` on the env's own refusals, an HTTP
error on transport / server faults).
"""

from __future__ import annotations

import asyncio
from typing import Any

import requests


class EnvClient:
    def __init__(self, url: str, timeout_s: float = 600.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def call_sync(self, name: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        """One verb; ``timeout`` (s) overrides the client's for this call (a slow SLAM render)."""
        resp = requests.post(f"{self.url}/{name}", json=kwargs, timeout=timeout or self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    async def call(self, name: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.call_sync, name, timeout, **kwargs)

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
