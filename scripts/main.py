"""Run the full IK -> plan -> control pipeline on one or more targets.

Flow:
    1. load params, seed RNGs, build the sim
    2. get the targets for the configured mode (fixed | blocked_pair | sequence | interactive)
    3. run them one by one, with or without the viewer
"""
import argparse
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import yaml

from src.sim_interface import SimInterface
from src.orchestrator import move_to_target
from src.kinematics import forward_kinematics
from src.transforms import homog_to_R_p, R_to_quat
from src.viz import draw_markers
from scripts.validate_planner import generate_blocked_pair
from utils.config import config as static_config

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "config" / "params.yaml"
START_KEY = 32     # GLFW keycode for SPACE
CONFIRM_KEY = 257  # GLFW keycode for ENTER
FRAME_DT = 0.01    # viewer refresh period while idle


# ============ Setup ============

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def load_params(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_rngs(seed):
    np.random.seed(seed)
    random.seed(seed)  # path smoothing samples shortcuts with the stdlib RNG


def build_move_kwargs(params):
    return dict(
        planner_kwargs=params["planner"],
        failure_retries=params["execution"]["failure_retries"],
        **params["controller"],
    )


# ============ Viewer ============

class ViewerSession:
    """Passive viewer plus the keys pressed in it (the key callback runs on the viewer thread)."""

    def __init__(self, viewer, pressed):
        self.viewer = viewer
        self._pressed = pressed

    @property
    def running(self):
        return self.viewer.is_running()

    def wait_for_key(self, key, message=None, on_frame=None):
        """Block until `key` is pressed; returns False if the viewer was closed first."""
        on_frame = on_frame or self.viewer.sync
        self._pressed.discard(key)
        if message:
            print(message)
        while self.running and key not in self._pressed:
            on_frame()
            time.sleep(FRAME_DT)
        return self.running

    def pause(self, seconds):
        t_end = time.time() + seconds
        while self.running and time.time() < t_end:
            self.viewer.sync()
            time.sleep(FRAME_DT)

    def hold_open(self):
        while self.running:
            self.viewer.sync()
            time.sleep(FRAME_DT)


@contextmanager
def open_viewer(sim):
    pressed = set()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=pressed.add) as viewer:
        yield ViewerSession(viewer, pressed)


# ============ Targets ============

@dataclass
class Target:
    name: str
    pos: np.ndarray
    quat: np.ndarray  # w, x, y, z


def fixed_targets(params):
    t = params["target"]
    return [Target("fixed target", np.array(t["pos"]), np.array(t["quat"]))]


def blocked_pair_targets(sim):
    """Random start/goal whose straight edge is blocked. Moves the sim to the start config."""
    q_start, q_goal = generate_blocked_pair(sim)
    sim.set_joint_angles(q_start)
    R_goal, pos_goal = homog_to_R_p(forward_kinematics(sim, q_goal)[0])
    return [Target("blocked pair goal", pos_goal, R_to_quat(R_goal))]


def sequence_targets(params):
    return [Target(t.get("name", f"target {i}"), np.array(t["pos"]), np.array(t["quat"]))
            for i, t in enumerate(params["target"]["sequence"])]


def interactive_targets(sim, session):
    """Yields a target each time ENTER is pressed; the user drags the "target" mocap body in between."""
    mocap_id = sim.model.body("target").mocapid[0]
    pos = sim.data.mocap_pos[mocap_id]
    quat = sim.data.mocap_quat[mocap_id]

    # Start the draggable target at the current end-effector pose
    R_ee, p_ee = homog_to_R_p(forward_kinematics(sim, sim.get_current_joint_angles_list())[0])
    pos[:] = p_ee
    quat[:] = R_to_quat(R_ee)

    def refresh():
        mujoco.mj_forward(sim.model, sim.data)  # update the mocap body's rendered pose
        draw_markers(session.viewer, pos, quat)

    n = 0
    while session.wait_for_key(
            CONFIRM_KEY, on_frame=refresh,
            message="\nDouble-click the green target to select it, then "
                    "Ctrl+Right-drag to move / Ctrl+Left-drag to rotate. Press ENTER to go."):
        n += 1
        yield Target(f"picked target {n}", pos.copy(), quat.copy())


def load_targets(sim, params):
    mode = params["target"]["mode"]
    if mode == "fixed":
        return fixed_targets(params)
    if mode == "blocked_pair":
        return blocked_pair_targets(sim)
    if mode == "sequence":
        return sequence_targets(params)
    raise ValueError(f"Unknown target mode: {mode}")


# ============ Execution ============

def run_targets(sim, targets, move_kwargs, session=None, pause=0.0, wait_for_start=False):
    """Move to each target in turn. A target that can't be reached is skipped.

    With a viewer, after each path is drawn: wait for SPACE (first target only, if
    `wait_for_start`), then hold for `pause` seconds so the plan is visible before moving.
    """
    viewer = session.viewer if session else None

    for i, target in enumerate(targets):
        if session and not session.running:
            break
        print(f"\n=== {target.name} ===")

        def before_execute(first=(i == 0)):
            if first and wait_for_start:
                session.wait_for_key(START_KEY, "Press SPACE in the viewer to start")
            session.pause(pause)

        q_start = np.array(sim.get_current_joint_angles_list())
        try:
            move_to_target(sim, target.pos, target.quat, viewer=viewer, q_start=q_start,
                           before_execute=before_execute if session else None, **move_kwargs)
        except RuntimeError as e:
            sim.set_joint_angles(q_start)
            print(f"Skipping {target.name}: {e}")


def main():
    args = parse_args()
    params = load_params(args.params)
    seed_rngs(params["seed"])

    mode = params["target"]["mode"]
    interactive = mode == "interactive"
    use_viewer = params.get("viewer", True) and not args.no_viewer
    if interactive and not use_viewer:
        raise ValueError("target.mode 'interactive' needs the viewer")

    sim = SimInterface(static_config[params["scene"]], add_target_mocap=interactive)
    move_kwargs = build_move_kwargs(params)

    # Resolve targets before the viewer opens (blocked_pair sampling moves the arm around)
    targets = None if interactive else load_targets(sim, params)

    if not use_viewer:
        run_targets(sim, targets, move_kwargs)
        return

    with open_viewer(sim) as session:
        if interactive:
            # ENTER already triggers each move, so no start wait / pause
            run_targets(sim, interactive_targets(sim, session), move_kwargs, session)
        else:
            run_targets(sim, targets, move_kwargs, session,
                        pause=params["target"].get("pause", 0.0),
                        wait_for_start=params.get("wait_for_start", False))
            print("\nDone. Close the viewer to exit.")
            session.hold_open()


if __name__ == "__main__":
    main()
