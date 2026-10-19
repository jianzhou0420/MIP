"""The MCP bridge every tool set runs (mip.core.toolset.ToolSet): spawned by the harness as a
stdio MCP server, one process per episode, the tools it registers named by ``MIP_BRIDGE``.

Run as a script by the harnesses (``python mip/core/bridge.py``), imported by path for the
recorded tool schemas (mip.core.panel.bridge_tool_schemas) — both under the episode's
environment, which is the only input.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":  # a script's folder is sys.path[0]; the tools import from the repo root
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from mip.core.toolset import SPEC_ENV, build_server

mcp = build_server(json.loads(os.environ[SPEC_ENV]))

if __name__ == "__main__":
    mcp.run(transport="stdio")
