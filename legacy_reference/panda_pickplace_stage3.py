"""Franka Panda Stage 3: physical Pick-and-Place with MuJoCo.

This extends the user's successful Stage 2 grasp-and-lift experiment.
Keep the following files in the same robot_simulation folder:
  panda_pickplace_stage1.py
  panda_pickplace_stage2_grasp.py
  panda_pickplace_stage3.py
  mujoco_menagerie/franka_emika_panda/{scene.xml,panda.xml,assets/...}

Run: python panda_pickplace_stage3.py

Sequence: approach -> descend -> close -> lift -> hold -> transfer
          -> place -> release -> retreat -> settle.
The cube has a real free joint and is moved only via finger contact.
"""

import csv
import json
import os
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from panda_pickplace_stage1 import (
    CUBE_HALF_WIDTH,
    CUBE_POS,
    PANDA_DIR,
    TABLE_POS,
    TABLE_SIZE,
    prepare_scene,
)
from panda_pickplace_stage2_grasp import (
    DAMPING,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HAND_TO_GRASP_LOCAL,
    JOINT_LIMIT_MARGIN,
    K_POS,
    K_ROT,
    MAX_JOINT_CMD_SPEED,
    ROOT,
    count_cube_contacts,
    mj_id,
    orientation_error_world,
    quintic,
)

# ============== Adjustable experiment parameters ==============
RESULTS = ROOT / "pickplace_stage3_results"
SHOW_VIEWER = True
REALTIME_WHEN_VISIBLE = True

# Same grasp preset already proven in Stage 2.
PREGRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.10
SETTLE_BEFORE_GRASP = 1.5

# New action: move the held cube +6 cm along world Y.
TRANSFER_OFFSET_XY = np.array([0.0, 0.06])  # meters

APPROACH_TIME = 2.5
DESCEND_TIME = 3.0
CLOSE_TIME = 2.0
LIFT_TIME = 3.0
HOLD_TIME = 0.8
TRANSFER_TIME = 3.0
PLACE_TIME = 3.0
RELEASE_TIME = 1.5
RETREAT_HEIGHT = 0.08
RETREAT_TIME = 2.0
FINAL_SETTLE_TIME = 1.2

# Physical and task-level verification conditions.
POSE_ABORT_MM = 12.0     # do not close fingers from a badly aligned hand pose
MIN_LIFT_MM = 40.0      # cube must lift at least 4 cm during lift stage
XY_TOLERANCE_MM = 20.0  # final cube center within 2 cm of desired XY
Z_TOLERANCE_MM = 10.0   # cube bottom within 1 cm of tabletop
STABLE_SPEED_M_S = 0.03
STABLE_DRIFT_MM = 10.0  # drift over last 0.6 s


def fmt(values):
    return " ".join(f"{float(x):.8g}" for x in values)


def make_stage3_scene():
    """Reuse successful Stage1 generator and add a green target visualization."""
    prepare_scene()  # writes derived panda_pickplace.xml and scene_pickplace.xml
    src = PANDA_DIR / "scene_pickplace.xml"
    dst = PANDA_DIR / "scene_pickplace_stage3.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Generated scene has no worldbody")
    target_xy = np.asarray(CUBE_POS[:2], dtype=float) + TRANSFER_OFFSET_XY
    table_top_z = TABLE_POS[2] + TABLE_SIZE[2]
    ET.SubElement(world, "site", {
        "name": "place_target_marker",
        "type": "sphere",
        "pos": fmt([target_xy[0], target_xy[1], table_top_z + 0.003]),
        "size": "0.008",
        "rgba": "0.1 0.9 0.3 0.8",
    })
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def main():
    scene = make_stage3_scene()
    # Use a relative path for the user's Windows directory with Chinese characters.
    os.chdir(ROOT)
    model = mujoco.MjModel.from_xml_path(
        "mujoco_menagerie/franka_emika_panda/" + scene.name
    )
    data = mujoco.MjData(model)

    home_id = mj_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    hand_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    left_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    left_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    right_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    cube_free_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    grip_actuator = mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

    joint_ids = np.array([
        mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 8)
    ], dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    qlo = np.maximum(model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[:7, 0])
    qhi = np.minimum(model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[:7, 1])
    qlo += JOINT_LIMIT_MARGIN
    qhi -= JOINT_LIMIT_MARGIN
    if grip_actuator != 7 or np.any(qlo >= qhi):
        raise RuntimeError("Unexpected Panda gripper or joint limits")

    finger_pos_ids = [
        int(model.jnt_qposadr[left_joint]),
        int(model.jnt_qposadr[right_joint]),
    ]
    cube_vel_id = int(model.jnt_dofadr[cube_free_joint])
    dt = float(model.opt.timestep)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    # geom_xpos is valid only after mj_forward; avoid using uninitialized zero poses.
    table_surface_z = float(data.geom_xpos[table_geom, 2] + model.geom_size[table_geom, 2])
    qcmd = np.clip(data.qpos[qpos_ids].copy(), qlo, qhi)
    data.ctrl[:7] = qcmd
    data.ctrl[grip_actuator] = GRIPPER_OPEN

    print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("Stage 3: grasp, lift, move world-Y by 6 cm, lower, release, retreat")
    print("Green marker: desired final cube XY; orange marker: initial cube XY")

    # Settle to tabletop in advance exactly as Stage 2 did.
    for _ in range(round(SETTLE_BEFORE_GRASP / dt)):
        data.ctrl[:7] = qcmd
        data.ctrl[grip_actuator] = GRIPPER_OPEN
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    cube_start = data.xpos[cube_body].copy()
    R_target = data.xmat[hand_id].reshape(3, 3).copy()
    down_alignment = float(R_target[:, 2] @ np.array([0.0, 0.0, -1.0]))
    if down_alignment < 0.90:
        raise RuntimeError(
            f"Gripper is not pointing down (alignment={down_alignment:.3f}); "
            "stop and configure wrist orientation first"
        )

    hand_grasp = cube_start - R_target @ HAND_TO_GRASP_LOCAL
    hand_pregrasp = hand_grasp + np.array([0.0, 0.0, PREGRASP_HEIGHT])
    hand_lift = hand_grasp + np.array([0.0, 0.0, LIFT_HEIGHT])
    transfer_shift = np.array([*TRANSFER_OFFSET_XY, 0.0])
    hand_transfer = hand_lift + transfer_shift
    hand_place = hand_grasp + transfer_shift
    hand_retreat = hand_place + np.array([0.0, 0.0, RETREAT_HEIGHT])
    target_cube_xy = cube_start[:2] + TRANSFER_OFFSET_XY

    print("Settled cube XYZ:", cube_start)
    print("Target cube XY:", target_cube_xy)
    print("Grasp hand XYZ:", hand_grasp)
    print("Transfer hand XYZ:", hand_transfer)
    print("Place hand XYZ:", hand_place)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    rows = []
    clip_steps = 0
    abort_reason = ""
    checks = {"lift": False, "transfer": False}

    def hand_pose():
        return data.xpos[hand_id].copy(), data.xmat[hand_id].reshape(3, 3).copy()

    def contacts():
        return count_cube_contacts(
            model, data, cube_geom, table_geom, left_body, right_body
        )

    def record(stage, target_p, command):
        hand_xyz, hand_R = hand_pose()
        cube_xyz = data.xpos[cube_body].copy()
        table_n, left_n, right_n = contacts()
        row = {
            "time_s": float(data.time),
            "stage": stage,
            "hand_x_m": float(hand_xyz[0]),
            "hand_y_m": float(hand_xyz[1]),
            "hand_z_m": float(hand_xyz[2]),
            "target_hand_x_m": float(target_p[0]),
            "target_hand_y_m": float(target_p[1]),
            "target_hand_z_m": float(target_p[2]),
            "hand_position_error_mm": float(1000 * np.linalg.norm(target_p - hand_xyz)),
            "hand_orientation_error_deg": float(np.rad2deg(np.linalg.norm(
                orientation_error_world(R_target, hand_R)
            ))),
            "cube_x_m": float(cube_xyz[0]),
            "cube_y_m": float(cube_xyz[1]),
            "cube_z_m": float(cube_xyz[2]),
            "cube_speed_m_s": float(np.linalg.norm(data.qvel[cube_vel_id:cube_vel_id + 3])),
            "cube_target_xy_error_mm": float(1000 * np.linalg.norm(
                cube_xyz[:2] - target_cube_xy
            )),
            "finger_opening_mm": float(1000 * sum(
                data.qpos[i] for i in finger_pos_ids
            )),
            "gripper_command": float(command),
            "cube_table_contacts": int(table_n),
            "cube_left_contacts": int(left_n),
            "cube_right_contacts": int(right_n),
        }
        # Store expert observation/action pairs for possible future BC work.
        for i in range(7):
            row[f"q{i+1}_rad"] = float(data.qpos[qpos_ids[i]])
            row[f"qd{i+1}_rad_s"] = float(data.qvel[dof_ids[i]])
            row[f"cmd{i+1}_rad"] = float(qcmd[i])
        rows.append(row)

    def execute(stage, target_p, T, command, viewer):
        nonlocal qcmd, clip_steps, abort_reason
        p0, _ = hand_pose()
        steps = max(1, round(T / dt))
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                abort_reason = "MuJoCo viewer closed"
                return False
            wall_start = time.perf_counter()
            t = step * dt
            s, sd = quintic(t, T)
            pdes = p0 + s * (target_p - p0)
            velocity_ff = sd * (target_p - p0)
            xyz, R_actual = hand_pose()
            pos_error = pdes - xyz
            rot_error = orientation_error_world(R_target, R_actual)
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            J = np.vstack((jacp[:, dof_ids], jacr[:, dof_ids]))
            desired_twist = np.hstack((velocity_ff + K_POS * pos_error, K_ROT * rot_error))
            dq = J.T @ np.linalg.solve(
                J @ J.T + DAMPING**2 * np.eye(6), desired_twist
            )
            peak = float(np.max(np.abs(dq)))
            if peak > MAX_JOINT_CMD_SPEED:
                dq *= MAX_JOINT_CMD_SPEED / peak
            proposed = qcmd + dq * dt
            if np.any((proposed < qlo) | (proposed > qhi)):
                clip_steps += 1
            qcmd = np.clip(proposed, qlo, qhi)
            data.ctrl[:7] = qcmd
            data.ctrl[grip_actuator] = command
            mujoco.mj_step(model, data)
            record(stage, pdes, command)
            if viewer is not None:
                viewer.sync()
                if REALTIME_WHEN_VISIBLE:
                    remaining = dt - (time.perf_counter() - wall_start)
                    if remaining > 0:
                        time.sleep(remaining)
        current_p, _ = hand_pose()
        hand_error_mm = 1000 * np.linalg.norm(target_p - current_p)
        cube_p = data.xpos[cube_body]
        table_n, left_n, right_n = contacts()
        print(
            f"{stage:>9} | hand error {hand_error_mm:5.2f} mm"
            f" | cube XYZ {np.round(cube_p, 4)}"
            f" | contacts (table/L/R) {table_n}/{left_n}/{right_n}"
        )
        return True

    context = (mujoco.viewer.launch_passive(model, data)
               if SHOW_VIEWER else nullcontext(None))
    completed = True
    with context as viewer:
        stages = [
            ("approach", hand_pregrasp, APPROACH_TIME, GRIPPER_OPEN),
            ("descend", hand_grasp, DESCEND_TIME, GRIPPER_OPEN),
            ("close", hand_grasp, CLOSE_TIME, GRIPPER_CLOSED),
            ("lift", hand_lift, LIFT_TIME, GRIPPER_CLOSED),
            ("hold", hand_lift, HOLD_TIME, GRIPPER_CLOSED),
            ("transfer", hand_transfer, TRANSFER_TIME, GRIPPER_CLOSED),
            ("place", hand_place, PLACE_TIME, GRIPPER_CLOSED),
            ("release", hand_place, RELEASE_TIME, GRIPPER_OPEN),
            ("retreat", hand_retreat, RETREAT_TIME, GRIPPER_OPEN),
            ("settle", hand_retreat, FINAL_SETTLE_TIME, GRIPPER_OPEN),
        ]
        for stage, target, T, grip in stages:
            if stage == "close":
                error_mm = float(1000 * np.linalg.norm(
                    hand_grasp - data.xpos[hand_id]
                ))
                if error_mm > POSE_ABORT_MM:
                    abort_reason = f"Grasp hand pose error {error_mm:.1f} mm > {POSE_ABORT_MM} mm"
                    completed = False
                    break
            if stage == "transfer" and not checks["lift"]:
                abort_reason = "Lift did not pass verification; refusing transfer"
                completed = False
                break
            if not execute(stage, target, T, grip, viewer):
                completed = False
                break
            if stage == "hold":
                table_n, left_n, right_n = contacts()
                rise_mm = 1000 * float(data.xpos[cube_body, 2] - cube_start[2])
                checks["lift"] = (
                    rise_mm >= MIN_LIFT_MM and table_n == 0
                    and left_n > 0 and right_n > 0
                )
                print(f"LIFT VERIFIED: {checks['lift']} (rise={rise_mm:.1f} mm)")
            if stage == "transfer":
                transfer_rows = [r for r in rows if r["stage"] == "transfer"]
                min_cube_z = min(r["cube_z_m"] for r in transfer_rows)
                max_table_contacts = max(r["cube_table_contacts"] for r in transfer_rows)
                moved_xy = float(np.linalg.norm(
                    data.xpos[cube_body, :2] - cube_start[:2]
                ))
                checks["transfer"] = (
                    min_cube_z - cube_start[2] >= MIN_LIFT_MM / 1000
                    and max_table_contacts == 0
                    and moved_xy > 0.035
                )
                print(
                    f"TRANSFER VERIFIED: {checks['transfer']} "
                    f"(horizontal cube move={moved_xy*1000:.1f} mm)"
                )
                if not checks["transfer"]:
                    abort_reason = "Cube slipped/fell during transfer; release skipped"
                    completed = False
                    break
            if stage == "place":
                cube_bottom = float(data.xpos[cube_body, 2] - CUBE_HALF_WIDTH)
                gap = cube_bottom - table_surface_z
                table_n, _, _ = contacts()
                print(f"Before release: cube bottom-table gap={gap*1000:.1f} mm, "
                      f"table contacts={table_n}")
                # Do not intentionally drop from a height greater than 15 mm.
                if gap > 0.015:
                    abort_reason = "Cube still >15 mm above table at placement; release skipped"
                    completed = False
                    break

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "pickplace_log.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    final_xyz = data.xpos[cube_body].copy()
    final_table, final_left, final_right = contacts()
    xy_error_mm = 1000 * float(np.linalg.norm(final_xyz[:2] - target_cube_xy))
    bottom_gap_mm = 1000 * float((final_xyz[2] - CUBE_HALF_WIDTH) - table_surface_z)
    final_speed = float(np.linalg.norm(data.qvel[cube_vel_id:cube_vel_id + 3]))
    settle_rows = [r for r in rows if r["stage"] == "settle"]
    last_window = [r for r in settle_rows
                   if r["time_s"] >= (rows[-1]["time_s"] - 0.6)] if rows else []
    drift_mm = float("inf")
    if len(last_window) >= 2:
        cube_pts = np.array([[r["cube_x_m"], r["cube_y_m"], r["cube_z_m"]]
                             for r in last_window])
        drift_mm = float(1000 * np.max(np.linalg.norm(
            cube_pts - cube_pts[0], axis=1
        )))

    placed = (
        completed and checks["lift"] and checks["transfer"]
        and xy_error_mm <= XY_TOLERANCE_MM
        and abs(bottom_gap_mm) <= Z_TOLERANCE_MM
        and final_table > 0
        and final_left == 0 and final_right == 0
        and final_speed <= STABLE_SPEED_M_S
        and drift_mm <= STABLE_DRIFT_MM
    )
    result = {
        "task_success": bool(placed),
        "sequence_completed": bool(completed),
        "abort_reason": abort_reason,
        "lift_verified": bool(checks["lift"]),
        "transfer_verified": bool(checks["transfer"]),
        "target_cube_xy_m": target_cube_xy.tolist(),
        "final_cube_xyz_m": final_xyz.tolist(),
        "final_xy_error_mm": xy_error_mm,
        "final_cube_bottom_table_gap_mm": bottom_gap_mm,
        "final_cube_speed_m_s": final_speed,
        "last_0p6s_cube_drift_mm": drift_mm if np.isfinite(drift_mm) else None,
        "final_table_contacts": int(final_table),
        "final_left_contacts": int(final_left),
        "final_right_contacts": int(final_right),
        "joint_command_clipping_steps": int(clip_steps),
    }
    summary_path = RESULTS / "stage3_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        times = np.array([r["time_s"] for r in rows])
        cube_x = np.array([r["cube_x_m"] for r in rows])
        cube_y = np.array([r["cube_y_m"] for r in rows])
        cube_z = np.array([r["cube_z_m"] for r in rows])
        opening = np.array([r["finger_opening_mm"] for r in rows])
        table_contact = np.array([r["cube_table_contacts"] for r in rows])
        lcontact = np.array([r["cube_left_contacts"] for r in rows])
        rcontact = np.array([r["cube_right_contacts"] for r in rows])
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        axes[0].plot(times, cube_z, label="Actual cube Z")
        axes[0].axhline(cube_start[2], ls="--", color="gray", label="Start cube Z")
        axes[0].set_ylabel("Cube Z (m)")
        axes[0].legend()
        axes[1].plot(times, cube_x, label="Cube X")
        axes[1].plot(times, cube_y, label="Cube Y")
        axes[1].axhline(target_cube_xy[1], ls="--", color="gray", label="Target Y")
        axes[1].set_ylabel("Cube XY (m)")
        axes[1].legend()
        axes[2].plot(times, opening)
        axes[2].set_ylabel("Finger opening (mm)")
        axes[3].plot(times, table_contact, label="Cube-table")
        axes[3].plot(times, lcontact, label="Cube-left")
        axes[3].plot(times, rcontact, label="Cube-right")
        axes[3].set_ylabel("Contact count")
        axes[3].set_xlabel("Simulation time (s)")
        axes[3].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = RESULTS / "pickplace_diagnostics.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    else:
        plot_path = None

    print("\n========== Stage 3 result ==========")
    print(f"Sequence completed: {completed}")
    if abort_reason:
        print(f"ABORT reason: {abort_reason}")
    print(f"Lift verified: {checks['lift']}")
    print(f"Transfer verified: {checks['transfer']}")
    print(f"Final cube XY error: {xy_error_mm:.2f} mm")
    print(f"Cube bottom-table gap: {bottom_gap_mm:.2f} mm")
    print(f"Final table / left / right contacts: {final_table} / {final_left} / {final_right}")
    print(f"Final cube speed: {final_speed:.4f} m/s")
    print(f"Cube drift in last 0.6s: {drift_mm:.2f} mm")
    print(f"Joint command clipping steps: {clip_steps}")
    print(f"Pick-and-Place succeeded: {placed}")
    print("CSV:", csv_path if rows else "No log")
    print("Summary:", summary_path)
    print("Plot:", plot_path)


if __name__ == "__main__":
    main()
