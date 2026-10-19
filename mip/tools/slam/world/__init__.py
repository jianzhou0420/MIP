"""The SLAM world — this repo's SLAM stack on EmbodiedScore-envs: ``stack`` (the STANDARD body
at micro quanta under the line's metric wrapper, every frame fed), ``_slam_env`` (SlamEnv: the
fed SLAM session, the maps, the instruments' reads), ``_slam`` / ``_slam_server`` (ORB-SLAM3 in
its container, which mounts this folder), ``_mapping`` (occupancy grids and floors),
``_map_render`` (the map images), ``_frontier`` / ``_plan`` (frontiers, A* routes), ``_geom``
(pose helpers) and ``_body`` (WorldBody: the habitat body SLAM is fed from)."""
