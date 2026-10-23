"""Panda Pick-and-Place Stage 2: approach, close gripper, and lift a cube.

Place beside panda_pickplace_stage1.py in robot_simulation/.
Run: python panda_pickplace_stage2_grasp.py

Stage 2 is a first physical-contact grasp experiment. It is NOT yet a
complete pick-and-place task; there is no transfer or placement stage.

Reuses the generated table/cube scene from Stage 1. Does not teleport or
weld the cube to the gripper. Uses 6D Jacobian-DLS and quintic trajectories.
"""

import csv
import os
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from panda_pickplace_stage1 import prepare_scene

# ===== Parameters for first contact/grasp trial =====
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "pickplace_stage2_results"
SHOW_VIEWER = True
REALTIME_WHEN_VISIBLE = True

# The Panda finger pad middle is ~10.3 cm along the hand body's LOCAL +Z.
# The home pose points local +Z downward. Verify it at startup.
HAND_TO_GRASP_LOCAL = np.array([0.0, 0.0, 0.1029])
PREGRASP_HEIGHT = 0.08    # pregrasp point is 8 cm above the cube center
LIFT_HEIGHT = 0.10        # lift target is +10 cm from grasp pose

APPROACH_TIME = 2.5
DESCEND_TIME = 3.0
CLOSE_TIME = 2.0
LIFT_TIME = 3.0
HOLD_TIME = 0.8
SETTLE_TIME = 1.5

K_POS = 2.5
K_ROT = 2.0
DAMPING = 0.05
MAX_JOINT_CMD_SPEED = 0.3     # rad/s
JOINT_LIMIT_MARGIN = 0.02

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


def mj_id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Object {name!r} not found in compiled model")
    return int(result)


def quintic(t, T):
    u = np.clip(t / T, 0.0, 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
    return float(s), float(sd)


def orientation_error_world(R_target, R_actual):
    return Rotation.from_matrix(R_target @ R_actual.T).as_rotvec()


def count_cube_contacts(model, data, cube_geom, table_geom, left_body, right_body):
    table = left = right = 0
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if cube_geom not in (a, b):
            continue
        other = b if a == cube_geom else a
        body_other = int(model.geom_bodyid[other])
        if other == table_geom:
            table += 1
        if body_other == left_body:
            left += 1
        if body_other == right_body:
            right += 1
    return table, left, right


def main():
    # Stage 1 generator has already been tested on the user's model.
    prepare_scene()
    os.chdir(ROOT)  # Avoid Windows non-ASCII absolute-path parsing issues.
    model = mujoco.MjModel.from_xml_path(
        "mujoco_menagerie/franka_emika_panda/scene_pickplace.xml"
    )
    data = mujoco.MjData(model)

    home = mj_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    hand = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    left_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    left_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    right_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    grip_act = mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

    joint_ids = np.array([
        mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)
    ])
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    qlo = np.maximum(model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[:7, 0])
    qhi = np.minimum(model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[:7, 1])
    qlo += JOINT_LIMIT_MARGIN
    qhi -= JOINT_LIMIT_MARGIN
    if np.any(qlo >= qhi) or grip_act != 7:
        raise RuntimeError("Unexpected Panda joint/actuator definition")

    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    qcmd = np.clip(data.qpos[qpos_ids].copy(), qlo, qhi)
    data.ctrl[:7] = qcmd
    data.ctrl[grip_act] = GRIPPER_OPEN
    finger_pos_ids = (
        int(model.jnt_qposadr[left_joint]),
        int(model.jnt_qposadr[right_joint]),
    )

    # Let the cube land and robot settle BEFORE computing targets.
    for _ in range(round(SETTLE_TIME / dt)):
        data.ctrl[:7] = qcmd
        data.ctrl[grip_act] = GRIPPER_OPEN
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    cube_start = data.xpos[cube_body].copy()
    p_initial = data.xpos[hand].copy()
    R_target = data.xmat[hand].reshape(3, 3).copy()
    finger_forward_world = R_target[:, 2]
    down_alignment = float(np.dot(finger_forward_world, [0, 0, -1]))
    print("Settled cube XYZ:", cube_start)
    print("Hand XYZ:", p_initial)
    print("Hand local +Z world direction:", finger_forward_world)
    print("Finger-down alignment:", round(down_alignment, 4))
    if down_alignment < 0.90:
        raise RuntimeError(
            "Hand is not pointing down in home configuration. "
            "Do not use this top-down grasp preset; adjust wrist orientation first."
        )

    # The 'hand' origin is ~10.3 cm away from the gripping surface.
    # To put the pads at cube center, compensate that offset in world coords.
    hand_grasp = cube_start - R_target @ HAND_TO_GRASP_LOCAL
    hand_pregrasp = hand_grasp + np.array([0., 0., PREGRASP_HEIGHT])
    hand_lift = hand_grasp + np.array([0., 0., LIFT_HEIGHT])
    print("Pregrasp hand XYZ:", hand_pregrasp)
    print("Grasp hand XYZ:   ", hand_grasp)
    print("Lift hand XYZ:    ", hand_lift)
    print("Cube should be grasped by real finger contact; no attachment is used")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    records = []
    clip_steps = 0

    def hand_pose():
        return (data.xpos[hand].copy(),
                data.xmat[hand].reshape(3, 3).copy())

    def record(stage, target_p, gripper):
        xyz, mat = hand_pose()
        cube_xyz = data.xpos[cube_body].copy()
        table, left, right = count_cube_contacts(
            model, data, cube_geom, table_geom, left_body, right_body
        )
        opening_mm = float(
            (data.qpos[finger_pos_ids[0]] + data.qpos[finger_pos_ids[1]]) * 1000
        )
        rec = [float(data.time), stage, *[float(x) for x in xyz],
               *[float(x) for x in target_p],
               float(np.linalg.norm(target_p - xyz) * 1000),
               float(np.rad2deg(np.linalg.norm(
                   orientation_error_world(R_target, mat)
               ))),
               *[float(x) for x in cube_xyz],
               opening_mm, float(gripper), table, left, right]
        records.append(rec)

    def execute(stage, target_p, T, gripper, viewer):
        nonlocal qcmd, clip_steps
        p0, R0 = hand_pose()
        if np.rad2deg(np.linalg.norm(
            orientation_error_world(R_target, R0)
        )) > 5.0:
            print(f"WARNING: start of {stage}, orientation offset >5 degrees")
        steps = max(1, round(T / dt))
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                return False
            start_wall = time.perf_counter()
            t = min(step * dt, T)
            s, sd = quintic(t, T)
            pdes = p0 + s * (target_p - p0)
            vff = sd * (target_p - p0)
            xyz, Rnow = hand_pose()
            ep = pdes - xyz
            er = orientation_error_world(R_target, Rnow)
            mujoco.mj_jacBody(model, data, jacp, jacr, hand)
            J = np.vstack((jacp[:, dof_ids], jacr[:, dof_ids]))
            twist = np.hstack((vff + K_POS * ep, K_ROT * er))
            A = J @ J.T + DAMPING**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, twist)
            peak = float(np.max(np.abs(dq)))
            if peak > MAX_JOINT_CMD_SPEED:
                dq *= MAX_JOINT_CMD_SPEED / peak
            proposed = qcmd + dq * dt
            if np.any((proposed < qlo) | (proposed > qhi)):
                clip_steps += 1
            qcmd = np.clip(proposed, qlo, qhi)
            data.ctrl[:7] = qcmd
            data.ctrl[grip_act] = gripper
            mujoco.mj_step(model, data)
            record(stage, pdes, gripper)
            if viewer is not None:
                viewer.sync()
                if REALTIME_WHEN_VISIBLE:
                    remain = dt - (time.perf_counter() - start_wall)
                    if remain > 0:
                        time.sleep(remain)
        xyz_end, _ = hand_pose()
        err_mm = 1000 * np.linalg.norm(target_p - xyz_end)
        print(f"{stage:>9}: final hand error = {err_mm:.2f} mm, "
              f"cube Z = {data.xpos[cube_body, 2]:.4f} m")
        return True

    context = (mujoco.viewer.launch_passive(model, data)
               if SHOW_VIEWER else nullcontext(None))
    with context as viewer:
        if not execute("approach", hand_pregrasp, APPROACH_TIME,
                       GRIPPER_OPEN, viewer):
            return
        if not execute("descend", hand_grasp, DESCEND_TIME,
                       GRIPPER_OPEN, viewer):
            return

        # Do not close if the palm/hand is far from the planned grip pose.
        grip_pose_error = float(
            1000 * np.linalg.norm(hand_grasp - data.xpos[hand])
        )
        if grip_pose_error > 10.0:
            print(f"ABORT: grasp pose error {grip_pose_error:.1f} mm > 10 mm")
            print("Check reachability, wrist orientation and joint limits")
            return

        # Hold the same hand pose while the fingers close.
        if not execute("close", hand_grasp, CLOSE_TIME,
                       GRIPPER_CLOSED, viewer):
            return
        close_table, close_left, close_right = count_cube_contacts(
            model, data, cube_geom, table_geom, left_body, right_body
        )
        print("After close: cube-table contacts", close_table,
              "; left/right finger contacts:", close_left, close_right)
        if not (close_left > 0 and close_right > 0):
            print("WARNING: no confirmed bilateral cube-finger contact. "
                  "Lift is still attempted for diagnosis, but grasp may fail.")

        if not execute("lift", hand_lift, LIFT_TIME,
                       GRIPPER_CLOSED, viewer):
            return
        if not execute("hold", hand_lift, HOLD_TIME,
                       GRIPPER_CLOSED, viewer):
            return

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_file = RESULTS / "grasp_log.csv"
    with csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "stage", "hand_x_m", "hand_y_m", "hand_z_m",
            "hand_target_x_m", "hand_target_y_m", "hand_target_z_m",
            "hand_pos_error_mm", "hand_ori_error_deg",
            "cube_x_m", "cube_y_m", "cube_z_m", "opening_mm",
            "gripper_command", "cube_table_contacts", "cube_left_contacts",
            "cube_right_contacts"
        ])
        writer.writerows(records)

    # Graph actual cube lifting and finger contacts.
    times = np.array([r[0] for r in records])
    zs = np.array([r[12] for r in records])
    openings = np.array([r[13] for r in records])
    left_contacts = np.array([r[16] for r in records])
    right_contacts = np.array([r[17] for r in records])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(times, zs, label="Cube center Z")
    axes[0].axhline(cube_start[2], color="gray", ls="--",
                    label="Cube before grasp")
    axes[0].set_ylabel("Cube Z (m)")
    axes[0].legend()
    axes[1].plot(times, openings)
    axes[1].set_ylabel("Finger opening (mm)")
    axes[2].plot(times, left_contacts, label="Cube-left finger")
    axes[2].plot(times, right_contacts, label="Cube-right finger")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("Contact count")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_file = RESULTS / "grasp_diagnostics.png"
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

    final_cube_z = float(data.xpos[cube_body, 2])
    table, left, right = count_cube_contacts(
        model, data, cube_geom, table_geom, left_body, right_body
    )
    raised = final_cube_z - cube_start[2]
    # Check that the cube rose and stayed off the table during the final hold.
    lift_success = (raised > 0.04 and table == 0)
    print("\n==== Stage 2 result ====")
    print(f"Cube height increase: {raised * 1000:.1f} mm")
    print(f"Final cube-table contacts: {table}")
    print(f"Final left/right finger contacts: {left}/{right}")
    print(f"Grasp-and-lift succeeded: {lift_success}")
    print(f"Joint command clipping steps: {clip_steps}")
    print("CSV:", csv_file)
    print("Plot:", plot_file)
    print("If lift failed: inspect pose alignment, two-sided contact and cube friction.")


if __name__ == "__main__":
    main()
