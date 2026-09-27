
# Integration script to run the entire planning and control pipeline
from src.rrt_star import rrt_star, extract_path
from src.path_smoothing import node_to_q_path, smooth_path
from src.sim_interface import SimInterface
from utils.config import config
from scripts.validate_planner import generate_blocked_pair
import numpy as np
import time
import mujoco

from src.kinematics import forward_kinematics
from src.ik_solver import run_ik_multistart
from src.controller import controller
from src.motion_planner import plan_path
from src.transforms import homog_to_R_p, R_to_quat
from src.q_validity_check import is_valid, check_collision, check_joint_limits
from src.viz import draw_markers, path_to_ee_positions

def get_valid_target_q(sim, target_pos, target_quat, max_attempts=5):
    for attempt in range(max_attempts):
        ik_result = run_ik_multistart(sim, target_pos, target_quat)
        q_goal = np.array(ik_result[0])
        converged = ik_result[3]

        if converged and is_valid(sim, q_goal):
            return q_goal

    return None  # never found a valid, reachable goal


def move_to_target(sim: SimInterface, target_pos, target_quat,
                    Kp_pos=380, Ki_pos=0.0, Kd_pos=0.0,
                    Kp_rot=300, Ki_rot=0.0, Kd_rot=0.0,
                    lam=0.01, max_iter=5000,
                    pos_tol=0.005, rot_tol=0.035,
                    pos_tol_intermediate=0.03, rot_tol_intermediate=0.15,
                    settle_window=0.1, settle_window_intermediate=0.0,
                    failure_retries=3,
                    planner_kwargs=None,
                    viewer=None, q_start=None, before_execute=None):

    planner_kwargs = planner_kwargs or {}

    if q_start is None:
        q_start = np.array(sim.get_current_joint_angles_list())


    draw_markers(viewer, target_pos, target_quat)

    q_goal = get_valid_target_q(sim, target_pos, target_quat)
    if q_goal is None:
        raise RuntimeError(f"No valid, reachable joint config found for target pos={target_pos}, quat={target_quat}")
    print(q_goal)

    path = plan_path(sim, q_start, q_goal, **planner_kwargs)
    if path is None:
        raise RuntimeError(f"No path found from q_start={q_start} to q_goal={q_goal}")
    sim.set_joint_angles(q_start)  # IK/collision checks leave qpos at the last checked config
    if viewer is not None:
        draw_markers(viewer, target_pos, target_quat, path_to_ee_positions(sim, path))

    if before_execute is not None:
        before_execute()  # e.g. wait for a keypress in the viewer before moving

    success = False

    for _ in range(failure_retries):

        for waypoint, q in enumerate(path):

            T_wp, _ = forward_kinematics(sim, q)
            wp_R, wp_pos = homog_to_R_p(T_wp)
            wp_quat = R_to_quat(wp_R)

            is_last = (waypoint == len(path) - 1)

            controller_result = controller(
                sim, wp_pos, wp_quat,
                Kp_pos=Kp_pos, Ki_pos=Ki_pos, Kd_pos=Kd_pos,
                Kp_rot=Kp_rot, Ki_rot=Ki_rot, Kd_rot=Kd_rot,
                lam=lam, max_iter=max_iter,
                pos_tol=pos_tol if is_last else pos_tol_intermediate,   # loose for intermediate, tight for final
                rot_tol=rot_tol if is_last else rot_tol_intermediate,
                settle_window=settle_window if is_last else settle_window_intermediate,  # don't require settling mid-path
                viewer=viewer
            )

            if not controller_result['is_converged']:
                print(f"Warning: did not converge for waypoint {waypoint}")
                print(f"Finding new path...")
                q_current = np.array(sim.get_current_joint_angles_list())
                path = plan_path(sim, q_current, q_goal, **planner_kwargs)
                if path is None:
                    raise RuntimeError(f"No path found from q_current={q_current} to q_goal={q_goal}")
                sim.set_joint_angles(q_current)
                if viewer is not None:
                    draw_markers(viewer, target_pos, target_quat, path_to_ee_positions(sim, path))
                break
            else:
                print(f"Reached waypoint {waypoint}")
                if waypoint == len(path) - 1:
                    success = True
                    print("TASK COMPLETE")

        if success:
            break

        
        
   
















