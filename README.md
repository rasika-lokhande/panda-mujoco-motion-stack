# panda-mujoco-motion-stack


*Complete kinematics, control and planning stack implemented from scratch on a Franka Emika Panda (7-DOF) in MuJoCo.*




**Problem Statement**: [docs/problem_statement.md](docs/problem_statement.md) 

**Summary report**: [docs/report.md](docs/report.md)

- Forward kinematics via an explicit transform chain, checked against MuJoCo's own site pose 
- Damped least-squares (Levenberg-Marquardt) IK over the Jacobian, with multi-start restarts 
- A task-space PID controller (position + orientation error, damped-pseudoinverse Jacobian) driving the arm to target poses, with gains swept and validated across distance bands 
- An RRT* motion planner over joint-space configurations, with shortcut-based path smoothing, validated over randomly sampled blocked start/goal pairs. In the full pipeline, the path ends at the exact IK goal (not just within RRT*'s `goal_tol`) whenever that last edge is collision-free 

> - No built-in IK, control, or motion-planning library calls are used for any of the four. 
> - The transform chain, the Jacobian, the control law, and the planner were implemented directly.

---

## Setup

```bash
pip install -e . # install dependencies
```

Run `python -m scripts.verify_setup` first to sanity-check the model loads and the viewer opens.

## Project structure

```
src/
  kinematics.py             # Forward Kinematics: Building transform chain from joint angles
  ik_solver.py              # Inverse Kinematics: Jacobian + damped least squares solver
  controller.py             # Task-space PID controller driving the arm to a target pose
  q_validity_check.py       # Joint-limit and collision checks
  rrt_star.py               # RRT* planner over joint-space configurations
  path_smoothing.py         # Shortcut-based path smoothing + a path-smoothness metric
  motion_planner.py         # plan_path(): RRT* + smoothing, retried until it succeeds
  orchestrator.py           # move_to_target(): full IK -> plan -> execute pipeline
  transforms.py             # Rotation/homogeneous-transform helper functions
  sim_interface.py          # Thin wrapper around the MuJoCo model/data for joint & pose access
  viz.py                    # Visual-only viewer markers: target pose + planned end-effector path
  config.yaml               # Model/scene paths, joint limits, velocity limits, result paths

utils/
  config.py                 # Loads src/config.yaml into a shared `config` dict

scripts/
  main.py                   # Runs the full pipeline end-to-end
  validate_fk.py            # Compares FK output against MuJoCo's ground-truth pose
  validate_ik.py            # Runs IK over random reachable targets, checks convergence
  validate_controller.py    # Runs the controller over targets at increasing distance bands
  validate_planner.py       # Runs RRT* over randomly sampled valid start/goal pairs, checks success rate
  plot_planner_results.py   # Plots planner success rate, convergence, path cost, and tree size vs. distance
  pid_tuning.py             # Sweeps Kp_pos/Kp_rot and plots convergence per target
  verify_setup.py           # Opens the MuJoCo viewer and steps through home/ready poses
  inspect_model.py          # Prints joint/body/site info for the loaded model
  config/
    params.yaml             # Params for scripts/main.py

models/franka_emika_panda/
  scene_easy.xml            # Single box obstacle
  scene_narrow.xml          # Two boxes with a narrow gap
  scene_shelf.xml           # Tall divider + two-slot shelf (demo scene)

results/                    # CSVs and plots produced by the scripts above
```

## Usage

Run any script as a module from the repo root.

```bash
python -m scripts.validate_fk
python -m scripts.validate_ik
python -m scripts.validate_controller
python -m scripts.validate_planner
python -m scripts.pid_tuning
```

Each writes its CSVs/plots into the matching `results/` subfolder.

Run the full pipeline (plan + execute to a target) with:

```bash
python -m scripts.main --params scripts/config/params.yaml
```

Target, scene, planner, and controller settings are all read from `scripts/config/params.yaml`; pass `--no-viewer` to run headless.

### Scenes

Set `scene` in `params.yaml` to a key from `src/config.yaml`:

| Key | Scene |
|---|---|
| `SCENE_EASY_PATH` | Single box obstacle in front of the robot |
| `SCENE_NARROW_PATH` | Two boxes with a narrow gap between them |
| `SCENE_SHELF_PATH` | Tall divider splitting the workspace + a two-slot shelf; crossing between zones forces multi-waypoint detours |

### Target modes

Set `target.mode` in `params.yaml`:

| Mode | What it does |
|---|---|
| `sequence` | Visits the targets listed under `target.sequence` in order (`name`, `pos`, `quat` as w, x, y, z). Unreachable targets are skipped. |
| `interactive` | Drag a target in the viewer and press ENTER to send the arm there; repeats until the viewer is closed. Viewer only. |
| `fixed` | Single target from `target.pos` / `target.quat`. |
| `blocked_pair` | Random valid start/goal pair whose straight-line edge is blocked. |

### Viewer

- **Markers** (visual only, no effect on collisions): red sphere + RGB axes = target pose; yellow spheres/lines = planned end-effector path, redrawn on replan.
- **SPACE** starts the motion when `wait_for_start: true`. The first path is planned and drawn first, then the run waits.
- `target.pause` holds each planned path on screen for that many seconds before moving.
- **Interactive mode**: double-click the green target to select it, then **Ctrl + right-drag** to move it and **Ctrl + left-drag** to rotate it. Press **ENTER** to go.

## Results

- **FK**: max position/orientation error against MuJoCo's own reported pose, across 100 random joint configurations ([results/fk_validation/](results/fk_validation/)).
- **IK**: convergence rate, iterations-to-converge, and final error distribution over reachable random targets ([results/ik_validation/](results/ik_validation/)).
- **Control**: convergence rate, settling time, and overshoot as a function of target distance from home ([results/controller_validation/](results/controller_validation/)), plus the Kp sweep used to pick working gains ([results/pid_tuning/](results/pid_tuning/)).
- **Planning**: RRT* success rate, iterations-to-converge, path cost, and tree size as a function of start-goal distance, over randomly sampled blocked configuration pairs ([results/planner_validation/](results/planner_validation/)).



---

**Known issues / TODO**: [docs/known_issues.md](docs/known_issues.md)

---
