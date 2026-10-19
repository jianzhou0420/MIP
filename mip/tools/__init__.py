"""tools — everything the agent can call. An experiment picks a set (``agent.tools.use``);
mip.core.toolset registers them behind one MCP bridge per episode.

    nav/        the discrete navigation surface: observe · step · answer
    waypoint/   the panorama + numbered waypoints: observe | observe_waypoints · goto · stop
    slam/       the SLAM instruments (get_pose · get_map · …) + SlamHook, the env-side SLAM
"""
