"""Franka Panda: 6D Cartesian quintic trajectory tracking vs static-target DLS.

Save this file beside your existing panda_reach.py in robot_simulation.
Dependencies: pip install mujoco numpy scipy matplotlib
Run: python panda_cartesian_trajectory.py

The controlled endpoint is the origin of the `hand` body (not fingertip center).
This example does NOT handle collision avoidance or actual object grasping.
"""

from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ========== 1. Experiment parameters ========== 
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
RESULT_DIR = ROOT / "cartesian_trajectory_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MOVE_TIME = 3.0    # Move over 3 s
HOLD_TIME = 2.0    # Hold final pose for 2 s
TOTAL_TIME = MOVE_TIME + HOLD_TIME

TARGET_TRANSLATION_M = np.array([0.02, 0.0, 0.02])  # +2cm X, +2cm Z
TARGET_WORLD_Z_ROT_DEG = 10.0

K_POS = 2.0          # position error feedback (1/s)
K_ROT = 2.0          # rotation error feedback (1/s)
DLS_DAMPING = 0.05
MAX_JOINT_SPEED = 0.5     # command speed, rad/s
JOINT_LIMIT_MARGIN = 0.02  # rad
USE_FEEDFORWARD = True   # recommended; False for a follow-up ablation
SHOW_VIEWER = True       # set True to watch both trials (slower)

# ========== 2. Trajectory mathematics ========== 
def quintic_scale(t: float, duration: float) -> tuple[float, float]:
    """Return s(t), ds/dt; zero endpoint velocity for 0<=t<=duration."""
    if t <= 0.0:
        return 0.0, 0.0
    if t >= duration:
        return 1.0, 0.0
    u = t / duration
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    s_dot = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    return float(s), float(s_dot)


def orientation_error_world(R_des: np.ndarray, R_act: np.ndarray) -> np.ndarray:
    """Rotation vector in world frame; matches MuJoCo rotational Jacobian."""
    return Rotation.from_matrix(R_des @ R_act.T).as_rotvec()


def reference_pose(
    mode: str, t: float,
    p0: np.ndarray, R0: np.ndarray,
    delta_p: np.ndarray, delta_rot_world: np.ndarray,
):
    """Desired position, desired orientation, feedforward linear/angular twist."""
    if mode == "step":
        # Original baseline: target jumps instantly to the final pose.
        s, s_dot = 1.0, 0.0
    elif mode == "quintic":
        s, s_dot = quintic_scale(t, MOVE_TIME)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    p_des = p0 + s * delta_p
    # R(s) = Exp(s * log(R_goal @ R0.T)) @ R0, world-frame rotation.
    R_des = Rotation.from_rotvec(s * delta_rot_world).as_matrix() @ R0
    v_ff = s_dot * delta_p
    w_ff = s_dot * delta_rot_world
    return p_des, R_des, v_ff, w_ff


# ========== 3. Load the known working model ========== 
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Missing model: {MODEL_PATH}\n"
        "Put this script in robot_simulation beside mujoco_menagerie."
    )

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
if home_id < 0 or hand_id < 0:
    raise ValueError("The model must contain a 'home' keyframe and 'hand' body")

joint_ids = []
qpos_ids = []
dof_ids = []
for i in range(1, 8):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
    if j < 0:
        raise ValueError(f"Cannot find joint{i}")
    joint_ids.append(j)
    qpos_ids.append(int(model.jnt_qposadr[j]))
    dof_ids.append(int(model.jnt_dofadr[j]))

joint_ids = np.array(joint_ids, dtype=int)
qpos_ids = np.array(qpos_ids, dtype=int)
dof_ids = np.array(dof_ids, dtype=int)

# This Menagerie model uses its first seven actuators for arm joint position.
if model.nu < 7:
    raise ValueError("Expected at least 7 position actuators")
if not np.array_equal(model.actuator_trnid[:7, 0], joint_ids):
    raise ValueError("First 7 actuators do not match joint1..joint7 in order")
if not np.all(model.jnt_limited[joint_ids]):
    raise ValueError("An arm joint is missing its position limits")
if not np.all(model.actuator_ctrllimited[:7]):
    raise ValueError("An arm actuator is missing its control limits")

joint_low = np.maximum(model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[:7, 0])
joint_high = np.minimum(model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[:7, 1])
low = joint_low + JOINT_LIMIT_MARGIN
high = joint_high - JOINT_LIMIT_MARGIN
if np.any(low >= high):
    raise ValueError("Inconsistent joint/control limits")


def reset_home():
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)


def get_hand_pose():
    p = data.xpos[hand_id].copy()
    R = data.xmat[hand_id].reshape(3, 3).copy()
    return p, R


reset_home()
p0, R0 = get_hand_pose()
p_goal = p0 + TARGET_TRANSLATION_M
rotvec_total = np.deg2rad(TARGET_WORLD_Z_ROT_DEG) * np.array([0.0, 0.0, 1.0])
R_goal = Rotation.from_rotvec(rotvec_total).as_matrix() @ R0

print("Starting XYZ (m):", p0)
print("Final XYZ (m):   ", p_goal)
print("Orientation goal: +", TARGET_WORLD_Z_ROT_DEG, "degrees around world Z")


# ========== 4. A single control experiment ========== 
def run_experiment(mode: str):
    reset_home()  # Fair comparison: exactly the same initial state.
    dt = float(model.opt.timestep)
    q_command = np.clip(data.qpos[qpos_ids].copy(), low, high)
    data.ctrl[:7] = q_command

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    records = []
    clipped_steps = 0
    peak_joint_cmd_speed = 0.0

    if SHOW_VIEWER:
        import mujoco.viewer
        viewer_cm = mujoco.viewer.launch_passive(model, data)
    else:
        viewer_cm = nullcontext(None)

    with viewer_cm as viewer:
        while data.time < TOTAL_TIME - 1e-12:
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()
            t = float(data.time)
            p_des, R_des, v_ff, w_ff = reference_pose(
                mode, t, p0, R0, TARGET_TRANSLATION_M, rotvec_total
            )
            p_act, R_act = get_hand_pose()
            e_pos = p_des - p_act
            e_rot = orientation_error_world(R_des, R_act)

            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            J = np.vstack((jacp[:, dof_ids], jacr[:, dof_ids]))

            # FF + FB 6D twist (world frame); step baseline FF is zero.
            twist_des = np.hstack((
                (v_ff if USE_FEEDFORWARD else 0.0) + K_POS * e_pos,
                (w_ff if USE_FEEDFORWARD else 0.0) + K_ROT * e_rot,
            ))
            mat = J @ J.T + DLS_DAMPING**2 * np.eye(6)
            qvel_command = J.T @ np.linalg.solve(mat, twist_des)

            # Scale all joint velocities together, preserving direction.
            max_abs = float(np.max(np.abs(qvel_command)))
            if max_abs > MAX_JOINT_SPEED:
                qvel_command *= MAX_JOINT_SPEED / max_abs
            peak_joint_cmd_speed = max(
                peak_joint_cmd_speed, float(np.max(np.abs(qvel_command)))
            )

            proposed = q_command + qvel_command * dt
            if np.any((proposed < low) | (proposed > high)):
                clipped_steps += 1
            q_command = np.clip(proposed, low, high)
            data.ctrl[:7] = q_command

            # Advance physics, then compare *time-aligned* desired and actual.
            mujoco.mj_step(model, data)
            t_new = float(data.time)
            p_act_new, R_act_new = get_hand_pose()
            p_des_new, R_des_new, _, _ = reference_pose(
                mode, t_new, p0, R0, TARGET_TRANSLATION_M, rotvec_total
            )

            track_pos_mm = np.linalg.norm(p_des_new - p_act_new) * 1000
            track_rot_deg = np.rad2deg(
                np.linalg.norm(orientation_error_world(R_des_new, R_act_new))
            )
            goal_pos_mm = np.linalg.norm(p_goal - p_act_new) * 1000
            goal_rot_deg = np.rad2deg(
                np.linalg.norm(orientation_error_world(R_goal, R_act_new))
            )

            records.append([
                t_new, *p_des_new, *p_act_new,
                track_pos_mm, track_rot_deg,
                goal_pos_mm, goal_rot_deg,
                float(np.max(np.abs(qvel_command))),
            ])

            if viewer is not None:
                viewer.sync()
                delay = dt - (time.perf_counter() - wall_start)
                if delay > 0:
                    time.sleep(delay)

    if not records:
        raise RuntimeError(f"No samples collected for {mode}")

    log = np.array(records, dtype=float)
    columns = [
        "time_s", "desired_x_m", "desired_y_m", "desired_z_m",
        "actual_x_m", "actual_y_m", "actual_z_m",
        "tracking_pos_error_mm", "tracking_ori_error_deg",
        "final_goal_pos_error_mm", "final_goal_ori_error_deg",
        "max_joint_cmd_speed_rad_s",
    ]
    with (RESULT_DIR / f"{mode}_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(records)

    move = log[:, 0] <= MOVE_TIME + 1e-9
    tail = log[:, 0] >= log[-1, 0] - 0.2
    summary = {
        "mode": mode,
        "motion_tracking_pos_rmse_mm": float(np.sqrt(np.mean(log[move, 7] ** 2))),
        "motion_tracking_ori_rmse_deg": float(np.sqrt(np.mean(log[move, 8] ** 2))),
        "final_pos_error_mm": float(log[-1, 9]),
        "final_ori_error_deg": float(log[-1, 10]),
        "last_0p2s_max_pos_error_mm": float(np.max(log[tail, 9])),
        "last_0p2s_max_ori_error_deg": float(np.max(log[tail, 10])),
        "peak_joint_cmd_speed_rad_s": peak_joint_cmd_speed,
        "command_limit_clip_steps": clipped_steps,
        "success": bool(
            np.max(log[tail, 9]) < 5.0 and np.max(log[tail, 10]) < 2.0
        ),
    }
    return log, summary


# ========== 5. Two experiments, same DLS gains ========== 
logs = {}
summaries = []
for mode in ["step", "quintic"]:
    log, summary = run_experiment(mode)
    logs[mode] = log
    summaries.append(summary)
    print(
        f"{mode:7} | final XYZ={summary['final_pos_error_mm']:.4f} mm"
        f" | final orientation={summary['final_ori_error_deg']:.4f} deg"
        f" | moving ref RMSE={summary['motion_tracking_pos_rmse_mm']:.3f} mm"
        f" | success={summary['success']}"
    )

with (RESULT_DIR / "summary.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)


# ========== 6. Plots ========== 
def save_comparison(col: int, ylabel: str, filename: str, title: str):
    plt.figure(figsize=(9, 5))
    for mode, log in logs.items():
        plt.plot(log[:, 0], log[:, col], label=mode)
    plt.axvline(MOVE_TIME, color="gray", linestyle="--", label="end of move")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


# Fairly comparable: both measure distance to the same final goal.
save_comparison(9, "Position error to final goal (mm)",
                "final_goal_position_error.png", "Distance to FINAL goal")
save_comparison(10, "Orientation error to final goal (deg)",
                "final_goal_orientation_error.png", "Orientation error to FINAL goal")

# Caution: these references differ during the movement period.
save_comparison(7, "Tracking position error (mm)",
                "reference_tracking_position_error.png", "Error relative to each method's reference")
save_comparison(8, "Tracking orientation error (deg)",
                "reference_tracking_orientation_error.png", "Orientation error relative to each reference")

# Only quintic has a moving desired pose. Display each Cartesian axis.
log = logs["quintic"]
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i, axis in enumerate("XYZ"):
    axes[i].plot(log[:, 0], log[:, i + 1], "--", label=f"{axis} desired")
    axes[i].plot(log[:, 0], log[:, i + 4], label=f"{axis} actual")
    axes[i].set_ylabel(f"{axis} (m)")
    axes[i].grid(alpha=0.35)
    axes[i].legend(loc="best")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Quintic 6D trajectory: XYZ tracking")
fig.tight_layout()
fig.savefig(RESULT_DIR / "quintic_xyz.png", dpi=180)
plt.close(fig)

print("\nResults saved in:", RESULT_DIR)
print("Note: moving-reference RMSE is NOT directly comparable between step and quintic.")
print("Compare final-goal error, settling and commanded motion as well.")
