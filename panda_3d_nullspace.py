"""Franka Panda: 6D pose DLS control with/without null-space optimization.

Run from the robot_simulation directory where mujoco_menagerie is located:
    python panda_6d_nullspace.py
Requirements:
    python -m pip install mujoco numpy scipy matplotlib

The controlled point is the origin of the 'hand' body, not the fingertip center.
This is a simulation learning project and is not a real-robot safety controller.
"""

import csv
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

# ========== 1. Parameters you can change ==========
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = "scene.xml"  # To test 0.5 kg later: "scene_payload.xml"
MODEL_REL_PATH = Path("mujoco_menagerie") / "franka_emika_panda" / MODEL_FILE
MODEL_PATH = PROJECT_DIR / MODEL_REL_PATH
OUTPUT_DIR = PROJECT_DIR / "pose3d_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 12.0  # seconds
MOVE_XYZ = np.array([0.02, 0.00, 0.02])  # X+2 cm, Z+2 cm
ROTATE_WORLD_Z_DEG = 10.0  # rotate hand orientation around world Z
KP_POS = 2.0     # 1/s
KP_ROT = 2.0     # 1/s
DLS_LAMBDA = 0.05
MAX_QVEL = 0.5   # rad/s, command speed cap
LIMIT_MARGIN = 0.02  # rad, extra margin for joint position commands
SHOW_VIEWER = True  # change to True to watch each run

# Only difference between trials is null-space gain.
TRIALS = {"pose_only": 0.0, "pose_nullspace": 0.6}

# ========== 2. Load model and IDs ==========
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Cannot find model: {MODEL_PATH}")

# Using a relative path can help with Chinese-character Windows directories.
# Make paths relative to this script's directory if necessary.
import os
os.chdir(PROJECT_DIR)
model = mujoco.MjModel.from_xml_path(str(MODEL_REL_PATH))
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
if home_id < 0 or hand_id < 0:
    raise ValueError("Model must have a 'home' keyframe and a 'hand' body")

joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = []
qpos_ids = []
dof_ids = []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    joint_ids.append(jid)
    qpos_ids.append(int(model.jnt_qposadr[jid]))
    dof_ids.append(int(model.jnt_dofadr[jid]))

joint_ids = np.array(joint_ids)
qpos_ids = np.array(qpos_ids)
dof_ids = np.array(dof_ids)

if model.nu < 7:
    raise ValueError("Expected at least seven position actuators")
if not np.all(model.jnt_limited[joint_ids]):
    raise ValueError("All seven arm joints must have position limits")
if not np.all(model.actuator_ctrllimited[:7]):
    raise ValueError("All seven arm actuators must have control limits")

joint_lo = np.maximum(
    model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[:7, 0]
)
joint_hi = np.minimum(
    model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[:7, 1]
)
low = joint_lo + LIMIT_MARGIN
high = joint_hi - LIMIT_MARGIN
if np.any(low >= high):
    raise ValueError("Joint limits are inconsistent")
q_center = (low + high) / 2.0
half_ranges = (high - low) / 2.0

def reset_home():
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)


def get_pose():
    position = data.xpos[hand_id].copy()
    # xmat is row-major 3x3 rotation matrix (hand -> world).
    rotation = data.xmat[hand_id].reshape(3, 3).copy()
    return position, rotation


def orientation_error_world(R_target, R_current):
    """World-frame rotation vector taking current orientation to target."""
    return Rotation.from_matrix(R_target @ R_current.T).as_rotvec()


reset_home()
start_xyz, start_R = get_pose()
target_xyz = start_xyz + MOVE_XYZ
target_R = (
    Rotation.from_euler("z", ROTATE_WORLD_Z_DEG, degrees=True).as_matrix()
    @ start_R
)

print("Initial hand XYZ:", start_xyz)
print("Target hand XYZ: ", target_xyz)
print(f"Target orientation change: world-Z +{ROTATE_WORLD_Z_DEG} degrees")

# ========== 3. Run one 6D pose trial ==========
def run_trial(name, null_gain):
    reset_home()
    dt = model.opt.timestep
    q_cmd = data.ctrl[:7].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    rows = []
    clip_steps = 0
    eye7 = np.eye(7)

    viewer_context = (
        mujoco.viewer.launch_passive(model, data)
        if SHOW_VIEWER else nullcontext(None)
    )
    with viewer_context as viewer:
        while data.time < DURATION:
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()

            xyz, R_current = get_pose()
            ep = target_xyz - xyz
            er = orientation_error_world(target_R, R_current)

            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            # 只控制末端XYZ位置，不控制姿态
            J = jacp[:, dof_ids]
            desired_twist = KP_POS * ep

            # Primary task: damped least squares (6D -> 7 joints).
            matrix = (J @ J.T + DLS_LAMBDA**2 * np.eye(3))
            dq_task = J.T @ np.linalg.solve(matrix, desired_twist)

            # Secondary task: move joints toward the middle of their limits.
            q_actual = data.qpos[qpos_ids].copy()
            # 归一化关节居中目标的负梯度
            dq_posture = (-null_gain * (q_actual - q_center) / (half_ranges ** 2))
            J_pinv = np.linalg.pinv(J, rcond=1e-4)
            N = eye7 - J_pinv @ J
            dq_null = N @ dq_posture
            # 每隔500个仿真步检查一次
            if null_gain > 0 and len(rows) % 500 == 0:
                print(
                    f"时间: {data.time:.2f} s | "
                    f"Jacobian rank: {np.linalg.matrix_rank(J)}"
                )
                print(
                    "主任务速度范数:",
                    np.linalg.norm(dq_task)
                )
                print(
                    "零空间速度范数:",
                    np.linalg.norm(dq_null)
                )
                print(
                    "零空间对末端速度的影响:",
                    np.linalg.norm(J @ dq_null)
                )
            dq = dq_task + dq_null

            # Uniform velocity scaling (keeps joint direction ratios).
            max_abs = np.max(np.abs(dq))
            if max_abs > MAX_QVEL:
                dq *= MAX_QVEL / max_abs

            proposed = q_cmd + dq * dt
            if np.any((proposed < low) | (proposed > high)):
                clip_steps += 1
            q_cmd = np.clip(proposed, low, high)
            data.ctrl[:7] = q_cmd

            mujoco.mj_step(model, data)
            xyz_after, R_after = get_pose()
            ep_after = target_xyz - xyz_after
            er_after = orientation_error_world(target_R, R_after)
            q_after = data.qpos[qpos_ids].copy()

            # This dimensionless value is smaller when joints are more centered.
            center_cost = np.linalg.norm((q_after - q_center) / half_ranges)
            rows.append([
                float(data.time),
                float(np.linalg.norm(ep_after) * 1000),
                float(np.linalg.norm(er_after) * 180 / np.pi),
                float(center_cost),
                *[float(q) for q in q_after],
            ])

            if viewer is not None:
                viewer.sync()
                wait = dt - (time.perf_counter() - wall_start)
                if wait > 0:
                    time.sleep(wait)

    if not rows:
        raise RuntimeError("Simulation produced no data")
    values = np.array(rows)
    data_path = OUTPUT_DIR / f"{name}.csv"
    with data_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_s", "position_error_mm", "orientation_error_deg", "joint_center_cost"]
            + [f"joint{i}_rad" for i in range(1, 8)]
        )
        writer.writerows(rows)

    tail = values[:, 0] >= values[-1, 0] - 0.2
    pos_e = values[:, 1]
    rot_e = values[:, 2]
    summary = {
        "mode": name,
        "null_gain": null_gain,
        "final_pos_mm": pos_e[-1],
        "final_ori_deg": rot_e[-1],
        "pos_rmse_mm": float(np.sqrt(np.mean(pos_e**2))),
        "ori_rmse_deg": float(np.sqrt(np.mean(rot_e**2))),
        "final_joint_center_cost": values[-1, 3],
        "command_limit_clip_steps": clip_steps,
        "position_success":bool(np.max(pos_e[tail]) <= 5.0),
        "orientation_within_2deg":bool(np.max(rot_e[tail]) <= 2.0),
    }
    return values, summary


# ========== 4. Compare DLS vs DLS + null-space ==========
curves = {}
summaries = []
for name, null_gain in TRIALS.items():
    values, summary = run_trial(name, null_gain)
    curves[name] = values
    summaries.append(summary)
    print(
        f"{name}: final XYZ={summary['final_pos_mm']:.4f} mm, "
        f"orientation={summary['final_ori_deg']:.4f} deg, "
        f"joint-center-cost={summary['final_joint_center_cost']:.4f}, "
        f"position_success={summary['position_success']}"
    )

with (OUTPUT_DIR / "summary.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

for column, ylabel, filename in [
    (1, "Position error (mm)", "position_error.png"),
    (2, "Orientation error (deg)", "orientation_error.png"),
    (3, "Joint-center cost (smaller is better)", "joint_center_cost.png"),
]:
    plt.figure(figsize=(9, 5))
    for name, values in curves.items():
        plt.plot(values[:, 0], values[:, column], label=name)
    plt.xlabel("Simulation time (s)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)

print("Saved outputs to:", OUTPUT_DIR)
plt.show()
