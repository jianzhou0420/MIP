"""The waypoint-predictor service the Waypoint, Hybrid and Hybrid + SLAM interfaces call (run.wp_server).
It runs in its own interpreter, ``.conda/ac-wp`` (requirements.txt here), started as a file — it does
not import the ``mip`` package:

    .conda/ac-wp/bin/python mip/tools/waypoint/predictor/server.py --port 9210

    GET  /health                             {"status": "ok", "name": "smartway_waypoint", "version": "2.0"}
    POST /call/smartway_waypoint__predict    {"inputs": {"views": [{"dir_id", "rgb_base64", "depth_base64"}, ...]}}
                                             -> {"outputs": {"candidates": {"0": {"angle", "distance"}, ...},
                                                             "num_candidates": K}}

Version 2.0 feeds each view's RGB through DINOv2, as SmartWay's predictor was trained; 1.0, the
predictor of the paper's Waypoint / Hybrid / Hybrid + SLAM runs, left the RGB embedding at zeros.
The checkpoints are read from ``<repo>/data/waypoint/`` (``--ckpt-dir``); the model loads before the
port opens, so a healthy /health means the predictor is ready. See engine.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

NAME = "smartway_waypoint"
VERSION = "2.0"
PREDICT_FN = "smartway_waypoint__predict"


def build_app(engine):
    from fastapi import FastAPI, HTTPException

    from engine import decode_views

    app = FastAPI(title="waypoint predictor")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "name": NAME, "version": VERSION}

    @app.post("/call/{function_name}")
    def call(function_name: str, body: dict) -> dict:
        if function_name != PREDICT_FN:
            raise HTTPException(404, f"Unknown function: {function_name}")
        inputs = body.get("inputs", body)
        candidates = engine.predict(*decode_views(inputs.get("views") or []))
        return {"outputs": {"candidates": candidates, "num_candidates": len(candidates)}}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9210)
    parser.add_argument("--ckpt-dir", type=Path, default=REPO / "data" / "waypoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from engine import WaypointEngine

    uvicorn.run(build_app(WaypointEngine(args.ckpt_dir)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
