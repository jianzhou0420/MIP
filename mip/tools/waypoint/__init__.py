"""The waypoint tools (the paper's Waypoint and Hybrid interfaces): a depth-based predictor
proposes candidate waypoints, they are drawn as numbered circles on a four-view panorama, and
the model picks one by number.

    observe             the panorama + numbered waypoints, as the Waypoint set's only look
    observe_waypoints   the same, beside the primitive observe/step (the Hybrid set)
    goto(waypoint)      walk to one candidate of the LATEST look (env verb step_hightolow)
    stop()              discrete action 0; ends the episode

Two servers: the ES env server, and the waypoint predictor service (``POST /call/{fn}`` with
``{"inputs": …}`` -> ``{"outputs": …}``; run.wp_server). Switches read: ``wp_max_moves`` (goto
cap; 0 = none), ``wp_predict_fn``, ``wp_view_px``, ``auto_observe`` (goto carries the new
panorama — the Waypoint set only), ``typed_results`` (see goto).

The family's ``preflight`` (mip.core.toolset.ToolSet.preflight) refuses a run without its predictor.
"""

from __future__ import annotations

from mip.tools.waypoint.panorama import preflight

__all__ = ["preflight"]
