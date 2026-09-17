"""FakeHarness — a token-free harness that plays a scripted tool sequence.

The whole runner path (placement → briefing → bridge spawn → session →
evaluate → record → callbacks) with no model: the "agent" is a fixed list
of tool calls over the arm's REAL bridge (stdio MCP, the same bridge_path
and bridge_env a real session gets). For wiring checks and A/B parity of
the runner itself, never a board seat.

    python runner.py <run name> +run.fake=true run.episodes=0 run.servers=[<url>]

The script comes from ``--set script='[["observe", {}], ["step", {"actions": [1, 2]}], ...]'``
or defaults to a short walk that ends the episode the way the task shape
does (STOP / answer / SUBTASK_STOP) — ``core.api.fake_wire.default_script``,
the same walk the fake ENDPOINT plays for a real harness
(``api=fake``). Records every call as tool_use / tool_result
events and one raw record per call; SessionOutcome carries no usage and no
cost.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from core.api.fake_wire import Script, default_script
from core.episode import EpisodeContext, SessionOutcome
from core.panel import json_safe


class FakeHarness:
    name = "fake"

    def __init__(self, *, script: Any = None) -> None:
        # configs/harness/fake.yaml: a default script (null = observe → step
        # → … derived from the bridge's tools); the run cfg's extra `script`
        # overrides it
        self.settings: dict[str, Any] = {"script": script}
        self.inherent = {
            "auth": "none (scripted)",
            "thinking": "none",
            "turn_cap": "script length",
            "vision_context": "none — images are received and discarded",
            "context_control": False,
            "context_label": "fake",
        }

    def prepare(self, spec: Any) -> None:
        return None

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {
            "options": {
                "script": ctx.extra.get("script") or self.settings["script"] or "default",
                "bridge": str(ctx.bridge_path),
            }
        }

    def _script(self, ctx: EpisodeContext, tools: dict[str, set[str]]) -> Script:
        raw = ctx.extra.get("script") or self.settings["script"]
        if not raw:
            return default_script(ctx.benchmark, tools)
        if isinstance(raw, str):
            raw = json.loads(raw)
        return [(str(name), dict(args or {})) for name, args in raw]

    async def run(self, ctx: EpisodeContext, sink: Any) -> SessionOutcome:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ctx.bridge_path)],
            env={**ctx.bridge_env(), "EH_EXECUTOR": "fake"},
        )
        n_calls = 0
        error: str | None = None
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = {
                    t.name: set((getattr(t, "inputSchema", None) or {}).get("properties") or {})
                    for t in listed.tools
                }
                sink.emit("system_init", {"model": "fake", "tools": sorted(tools)})
                sink.emit("bridge_status", {"status": "connected"})
                script = self._script(ctx, tools)
                for i, (tool, args) in enumerate(script):
                    if tool not in tools:
                        sink.emit("driver_error", {"error": f"script names unknown tool {tool!r}"})
                        continue
                    call_id = f"fake-{ctx.index}-{i}"
                    sink.emit(
                        "tool_use", {"id": call_id, "name": f"mcp__env__{tool}", "input": args}
                    )
                    result = await session.call_tool(tool, args)
                    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
                    n_images = sum(1 for c in result.content if getattr(c, "type", "") == "image")
                    sink.emit(
                        "tool_result", {"tool_use_id": call_id, "texts": texts, "images": n_images}
                    )
                    sink.raw(
                        {
                            "type": "FakeCall",
                            "msg": json_safe(
                                {"tool": tool, "input": args, "texts": texts, "images": n_images}
                            ),
                        }
                    )
                    n_calls += 1
                    parsed = (
                        sink.parse_step_result(texts)
                        if hasattr(sink, "parse_step_result")
                        else None
                    )
                    if parsed and parsed.get("episode_over"):
                        break
                sink.emit("assistant_text", {"text": f"script complete ({n_calls} calls)"})
        return SessionOutcome(
            usage=None, cost_usd=None, turns=n_calls, error=error, extra={"script_calls": n_calls}
        )
