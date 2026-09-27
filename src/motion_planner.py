from src.rrt_star import rrt_star, extract_path
from src.path_smoothing import node_to_q_path, smooth_path, path_smoothness_metric
from src.sim_interface import SimInterface
from utils.config import config
from scripts.validate_planner import generate_blocked_pair
import numpy as np
import time
import mujoco

from src.kinematics import forward_kinematics
from src.controller import controller
from src.transforms import homog_to_R_p, R_to_quat
from src.q_validity_check import check_edge_validity





def plan_path(sim, q_start, q_goal, max_retries=5, smoothing: bool = True, rrt_kwargs=None):

    rrt_kwargs = rrt_kwargs or {}

    for _ in range(max_retries):
        rrt_star_result = rrt_star(sim, q_start, q_goal, **rrt_kwargs)
        if rrt_star_result['success']:
            break
        print("Trying again...")
    else:
        print("Path not found")
        return None

    node_path, _ = extract_path(rrt_star_result['end_node'])

    q_path = node_to_q_path(node_path)

    # RRT* stops within goal_tol of q_goal; finish at q_goal exactly when that last edge is free
    if not np.allclose(q_path[-1], q_goal) and check_edge_validity(sim, q_path[-1], q_goal):
        q_path = list(q_path) + [np.array(q_goal)]
    print(path_smoothness_metric(q_path))

    if smoothing:
        path = smooth_path(sim, q_path)
        print(path_smoothness_metric(q_path))
    else:
        path = q_path

    return path



if __name__ == '__main__':
    np.random.seed(42)


    ntrials = 5
    sim = SimInterface(config['SCENE_EASY_PATH'])


    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:


        for trial in range(ntrials):

            q_start, q_goal = generate_blocked_pair(sim)
            # print(q_start)
            # print(q_goal)
            path = plan_path(sim, q_start, q_goal)

            print(len(path))



            for wp, q in enumerate(path):
                T_ee, _ = forward_kinematics(sim, q)
                target_ee_R, target_ee_pos = homog_to_R_p(T_ee)
                target_ee_quat = R_to_quat(target_ee_R)

                controller_result = controller(sim, target_ee_pos, target_ee_quat, Kp_pos = 380, Kp_rot=300, viewer=viewer)

                
                # import matplotlib.pyplot as plt
                # plt.ion()  # turn on interactive mode
                # plt.plot(controller_result['pos_err_history'], label='pos error')
                # plt.axhline(y=0.005, color='r', linestyle='--', label='pos_tol')
                # plt.xlabel('iteration')
                # plt.ylabel('error')
                # plt.legend()
                # plt.title(f'Waypoint {wp} convergence')
                # plt.show()


                if not controller_result['is_converged']:
                    print(f"Warning: did not converge for waypoint {wp}")
                    break
                    
                else:
                    print(f"Converged for waypoint {wp}")



        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

    

