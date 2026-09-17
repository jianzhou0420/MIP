"""ui.server — the run monitor: a read-only view over ``outputs/``.

    python -m ui.server [--port 8130] [--outputs <dir>]     ->  http://10.12.65.19:8130 (this box's LAN address)

Nothing in the runner knows this exists. The page polls a handful of JSON
routes, and every route is a fresh read of the run directory the runner
writes (summary.json flushed per episode, episode_{i}.jsonl per event,
live_{i}/obs_*.png per observation, stats.html at the end):

    GET /api/runs                                   every run under every output root
    GET /api/runs/<root>/<run>/summary              config + per-episode outcomes + honest aggregate
    GET /api/runs/<root>/<run>/episode/<i>/log?offset=N   event lines (incremental)
    GET /api/runs/<root>/<run>/episode/<i>/frames   frame names on disk
    GET /api/runs/<root>/<run>/episode/<i>/frame/<name>   the image
    GET /api/runs/<root>/<run>/file/<name>          stats.html · config.yaml · env_server.log · summary.json

Scoring rule for the aggregate (frozen 2026-08-01): an episode is SCORED iff
``metrics.success`` exists and it is not a limit-exceeded casualty (error
AND zero env steps — never attempted, excluded, shown ⚠); turn-exhaustion
navigated and counts as a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import mimetypes
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.paths import OUTPUT_ROOTS  # noqa: E402  (after the sys.path line)

DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
INDEX = HERE / "index.html"
SAFE = re.compile(r"^[A-Za-z0-9._\-]+$")
FILES = ("stats.html", "config.yaml", "env_server.log", "summary.json", "code_state.json")
LIVE_WINDOW_S = 90.0  # a run whose files moved within this window is shown as live


# ── scoring (the monitor's own, never the driver's stored aggregate) ──


def is_limit_exceeded(ep: dict[str, Any]) -> bool:
    return bool(ep.get("error")) and not ((ep.get("agent") or {}).get("env_steps") or 0)


def display_aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        e for e in episodes
        if (e.get("metrics") or {}).get("success") is not None and not is_limit_exceeded(e)
    ]
    agg: dict[str, Any] = {"episode_count": len(scored)}
    if not scored:
        return agg
    for key in ("success", "spl", "ndtw", "oracle_success", "distance_to_goal"):
        vals = [float((e.get("metrics") or {})[key]) for e in scored
                if isinstance((e.get("metrics") or {}).get(key), (int, float))]
        if vals:
            agg[key] = round(sum(vals) / len(vals), 4)
    agg["stop_rate"] = round(
        sum(1 for e in scored if (e.get("agent") or {}).get("called_stop")) / len(scored), 4)
    return agg


# ── reads ──


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # mid-write or absent — the next poll gets it


def _episode_indices(run_dir: Path) -> list[int]:
    out = []
    for p in run_dir.glob("episode_*.jsonl"):
        tail = p.stem.split("_", 1)[1]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(out)


def _latest_mtime(run_dir: Path) -> float:
    latest = run_dir.stat().st_mtime
    for p in [*run_dir.glob("*"), *run_dir.glob("live_*/*")]:
        with contextlib.suppress(OSError):  # a file removed between glob and stat
            latest = max(latest, p.stat().st_mtime)
    return latest


def _run_entry(root: str, d: Path) -> dict[str, Any] | None:
    eps = _episode_indices(d)
    summary = _json(d / "summary.json") if (d / "summary.json").exists() else None
    if not eps and summary is None:
        return None
    mtime = _latest_mtime(d)
    entry: dict[str, Any] = {
        "name": d.name, "root": root, "mtime": mtime, "episodes": eps,
        "live": (time.time() - mtime) < LIVE_WINDOW_S,
    }
    if summary:
        agg = display_aggregate(summary.get("episodes") or [])
        cfg = summary.get("config") or {}
        entry.update(
            success=agg.get("success"), episode_count=agg.get("episode_count"),
            model=cfg.get("model") or (summary.get("provenance") or {}).get("model"),
            harness=summary.get("harness"),
            dataset=cfg.get("dataset"), split=cfg.get("split"),
        )
    return entry


class Monitor:
    def __init__(self, outputs: Path) -> None:
        self.outputs = outputs

    def roots(self) -> list[str]:
        """The harness axis, in a fixed order: one root per harness
        (core.paths.OUTPUT_ROOTS: claudecode · codex · mini-swe-agent · fake),
        shown whether or not it holds a run yet. Nothing else under outputs/
        (gateway logs, ui.log, logs/) is a root."""
        return [OUTPUT_ROOTS[k].name for k in ("cc", "codex", "mini", "fake")]

    def run_dir(self, root: str, run: str) -> Path:
        if not (SAFE.match(root) and SAFE.match(run)):
            raise ValueError("bad run path")
        d = self.outputs / root / run
        if not d.is_dir():
            raise FileNotFoundError(run)
        return d

    def runs(self) -> dict[str, Any]:
        runs = []
        for root in self.roots():
            if not (self.outputs / root).is_dir():
                continue  # a harness with no run yet
            for d in (self.outputs / root).iterdir():
                if d.is_dir():
                    e = _run_entry(root, d)
                    if e:
                        runs.append(e)
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return {"roots": self.roots(), "runs": runs}

    def summary(self, root: str, run: str) -> dict[str, Any]:
        d = self.run_dir(root, run)
        started = _episode_indices(d)
        data = _json(d / "summary.json") or {}
        eps = []
        for e in data.get("episodes") or []:
            m, a = e.get("metrics") or {}, e.get("agent") or {}
            eps.append({
                "index": e.get("index"), "episode_id": e.get("episode_id"),
                "instruction": e.get("instruction"),
                "success": m.get("success"), "spl": m.get("spl"),
                "distance_to_goal": m.get("distance_to_goal"),
                "env_steps": a.get("env_steps"), "called_stop": a.get("called_stop"),
                "end_reason": a.get("end_reason"), "num_turns": a.get("num_turns"),
                "cost_usd": (a.get("cost") or {}).get("usd"), "wall_sec": e.get("wall_sec"),
                "error": e.get("error"), "limit_exceeded": is_limit_exceeded(e),
            })
        cfg = data.get("config") or {}
        return {
            "run_name": run, "root": root, "started_episodes": started, "episodes": eps,
            "aggregate": display_aggregate(data.get("episodes") or []) if data else None,
            "config": {k: cfg.get(k) for k in ("dataset", "split", "episodes", "max_turns",
                                                "step_budget", "model", "arm", "harness")},
            "harness": data.get("harness"), "provenance": data.get("provenance"),
            "cost": data.get("cost"), "run_stats": data.get("run_stats"),
            "files": [f for f in FILES if (d / f).exists()],
            "live": (time.time() - _latest_mtime(d)) < LIVE_WINDOW_S,
        }

    def log(self, root: str, run: str, index: int, offset: int) -> dict[str, Any]:
        p = self.run_dir(root, run) / f"episode_{index}.jsonl"
        if not p.exists():
            return {"lines": [], "next_offset": offset}
        raw = p.read_text().splitlines()
        lines = []
        for line in raw[offset:]:
            try:
                lines.append(json.loads(line))
            except ValueError:
                continue  # a partial last line — re-read next poll
        return {"lines": lines, "next_offset": offset + len(lines)}

    def frames(self, root: str, run: str, index: int) -> dict[str, Any]:
        live = self.run_dir(root, run) / f"live_{index}"
        if not live.is_dir():
            return {"frames": []}
        # obs_ is a camera frame, map_ is a render of the occupancy map at that
        # moment. Both belong in the strip: a run where you can read the model's
        # words and see its pictures but never its MAP is missing the one
        # artefact a navigation run is actually about.
        names = sorted(p.name for p in live.iterdir()
                       if p.name.startswith(("obs_", "map_"))
                       and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        return {"frames": names}

    def frame_path(self, root: str, run: str, index: int, name: str) -> Path:
        if not SAFE.match(name):
            raise ValueError("bad frame name")
        return self.run_dir(root, run) / f"live_{index}" / name

    def file_path(self, root: str, run: str, name: str) -> Path:
        if name not in FILES:
            raise ValueError("not a served file")
        return self.run_dir(root, run) / name


# ── http ──


def make_handler(mon: Monitor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # polling would flood stdout

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _file(self, path: Path) -> None:
            if not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in (".log", ".yaml", ".json"):
                ctype = "text/plain; charset=utf-8"
            self._send(200, path.read_bytes(), ctype)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            parts = [p for p in url.path.split("/") if p]
            try:
                if not parts:
                    self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                elif parts[:2] == ["api", "runs"]:
                    rest = parts[2:]
                    if not rest:
                        self._json(mon.runs())
                    elif len(rest) == 3 and rest[2] == "summary":
                        self._json(mon.summary(rest[0], rest[1]))
                    elif len(rest) == 4 and rest[2] == "file":
                        self._file(mon.file_path(rest[0], rest[1], rest[3]))
                    elif len(rest) >= 5 and rest[2] == "episode":
                        i = int(rest[3])
                        if rest[4] == "log":
                            self._json(mon.log(rest[0], rest[1], i, int(q.get("offset", ["0"])[0])))
                        elif rest[4] == "frames":
                            self._json(mon.frames(rest[0], rest[1], i))
                        elif rest[4] == "frame" and len(rest) == 6:
                            self._file(mon.frame_path(rest[0], rest[1], i, rest[5]))
                        else:
                            self._json({"error": "no such route"}, 404)
                    else:
                        self._json({"error": "no such route"}, 404)
                else:
                    self._json({"error": "no such route"}, 404)
            except FileNotFoundError as exc:
                self._json({"error": f"no run {exc}"}, 404)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)

    return Handler


def lan_ip() -> str:
    """The address to open in a browser on the LAN — the interface that
    routes out, never the loopback."""
    import socket

    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    return socket.gethostbyname(socket.gethostname())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    args = parser.parse_args(argv)
    mon = Monitor(Path(args.outputs).resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(mon))
    print(f"[ui] monitoring {mon.outputs} on http://{lan_ip()}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
