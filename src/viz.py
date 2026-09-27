# Visual-only markers drawn into the passive viewer's user scene (no effect on physics/collisions)
import mujoco
import numpy as np

from src.kinematics import forward_kinematics
from src.transforms import quat_to_R, homog_to_R_p

TARGET_RGBA = np.array([1.0, 0.2, 0.2, 0.5], dtype=np.float32)
PATH_RGBA = np.array([1.0, 0.8, 0.0, 0.8], dtype=np.float32)
AXIS_RGBA = [np.array([1, 0, 0, 1], dtype=np.float32),
             np.array([0, 1, 0, 1], dtype=np.float32),
             np.array([0, 0, 1, 1], dtype=np.float32)]


def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def _add_connector(scn, geom_type, width, p_from, p_to, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, geom_type, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(geom, geom_type, width,
                         np.asarray(p_from, dtype=float), np.asarray(p_to, dtype=float))
    scn.ngeom += 1


def path_to_ee_positions(sim, path):
    positions = []
    for q in path:
        T, _ = forward_kinematics(sim, q)
        _, p = homog_to_R_p(T)
        positions.append(p)
    return positions


def draw_markers(viewer, target_pos, target_quat, path_positions=None,
                 target_radius=0.02, axis_len=0.08, waypoint_radius=0.008):
    """Redraw the target marker (sphere + RGB orientation axes) and, optionally, the planned EE path."""
    if viewer is None:
        return

    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0

        # Target: translucent sphere + orientation axes (x=red, y=green, z=blue)
        _add_sphere(scn, target_pos, target_radius, TARGET_RGBA)
        R = quat_to_R(target_quat)
        for i in range(3):
            _add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.004,
                           target_pos, np.asarray(target_pos) + axis_len * R[:, i], AXIS_RGBA[i])

        # Planned path: waypoint spheres joined by line segments
        if path_positions is not None:
            for i, p in enumerate(path_positions):
                _add_sphere(scn, p, waypoint_radius, PATH_RGBA)
                if i > 0:
                    _add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2.0,
                                   path_positions[i - 1], p, PATH_RGBA)

    viewer.sync()
