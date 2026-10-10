"""Workcell presentation events; these do not change robot or sensor state."""


def select_task_camera_view(env, env_ids=None):
    """Use the existing sensor camera in the GUI instead of an orbit camera.

    Headless runs have no viewport and require no presentation changes.
    """
    # Newton's GL camera is world-space. Offset by env_0's origin so a grid
    # centered elsewhere does not leave the camera looking between workcells.
    for visualizer in env.sim.visualizers:
        if visualizer.cfg.visualizer_type in ("newton", "newton_gl") and not getattr(
            visualizer, "_task_camera_selected", False
        ):
            origin = env.scene.env_origins[0]
            if hasattr(origin, "torch"):
                origin = origin.torch
            origin = origin.tolist()
            cfg = env.cfg.sim.default_visualizer_cfg
            visualizer.set_camera_view(
                tuple(a + b for a, b in zip(origin, cfg.eye, strict=True)),
                tuple(a + b for a, b in zip(origin, cfg.lookat, strict=True)),
            )
            visualizer._task_camera_selected = True
    if not env.sim.has_gui:
        return
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        return
    camera_path = env.scene.env_prim_paths[0] + "/Camera"
    if not env.sim.stage.GetPrimAtPath(camera_path).IsValid():
        raise RuntimeError(f"Task camera does not exist: {camera_path}")
    viewport.set_active_camera(camera_path)
