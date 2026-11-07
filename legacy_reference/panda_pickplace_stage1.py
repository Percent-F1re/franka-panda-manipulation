"""Pick-and-Place Stage 1: table + free cube + Panda gripper open/close.

Place this file in robot_simulation/, next to mujoco_menagerie/.
Run: python panda_pickplace_stage1.py

Stage 1 intentionally DOES NOT approach or grasp the cube; the gripper opens
and closes above the table while the cube falls onto the tabletop.  This
isolates model loading, the free body, finger actuators, and contact logging.

The script creates two new XML files in franka_emika_panda/ without modifying
panda.xml or scene.xml.  A derived Panda XML is necessary because adding a
freejoint adds 7 qpos elements to the model, so the original 'home' keyframe
must grow by 7 qpos values to compile correctly.
"""

import csv
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# =============== 0. User-adjustable options ===============
ROOT = Path(__file__).resolve().parent
PANDA_DIR = ROOT / "mujoco_menagerie" / "franka_emika_panda"
SOURCE_PANDA = PANDA_DIR / "panda.xml"
SOURCE_SCENE = PANDA_DIR / "scene.xml"
GENERATED_PANDA = PANDA_DIR / "panda_pickplace.xml"
GENERATED_SCENE = PANDA_DIR / "scene_pickplace.xml"
RESULTS = ROOT / "pickplace_stage1_results"

SHOW_VIEWER = True             # False: run without showing the 3D window
REALTIME_WHEN_VISIBLE = True   # Allows you to watch the finger motion
TOTAL_TIME = 5.5              # Simulation seconds

# Table height: tabletop = table center z + half thickness = 0.380 m
TABLE_POS = (0.60, 0.0, 0.345)
TABLE_SIZE = (0.30, 0.28, 0.035)

# The 4-cm-wide cube starts 4 cm above the tabletop, then falls onto it.
CUBE_POS = (0.58, 0.0, 0.44)
CUBE_HALF_WIDTH = 0.020
CUBE_MASS_KG = 0.080

# (Start time, gripper setpoint): Open -> Close -> Open
# Official Menagerie Panda gripper actuator8: 0 closed; 255 fully open.
GRIPPER_SCHEDULE = (
    (0.0, 255.0),
    (1.2, 0.0),
    (3.0, 255.0),
)


def fmt_xyz(values):
    return " ".join(f"{v:.8g}" for v in values)


def prepare_scene():
    """Make *copies* based on the user's existing Panda and scene XML."""
    for p in [SOURCE_PANDA, SOURCE_SCENE]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Put this script in the robot_simulation folder."
            )

    # 1) Clone the official robot file, only extending the keyframe for cube.
    panda_tree = ET.parse(SOURCE_PANDA)
    panda_root = panda_tree.getroot()
    home = panda_root.find("./keyframe/key[@name='home']")
    if home is None or not home.get("qpos"):
        raise ValueError("Cannot find Panda home keyframe with qpos in panda.xml")
    panda_home_qpos = home.get("qpos").split()
    if len(panda_home_qpos) != 9:
        raise ValueError(
            "Expected Panda home qpos to have 9 values (7 arm + 2 finger); "
            f"found {len(panda_home_qpos)}. Please check which panda.xml is used."
        )
    # Newly added free joint qpos = (x,y,z, quaternion w,x,y,z).
    cube_free_qpos = [*CUBE_POS, 1.0, 0.0, 0.0, 0.0]
    home.set("qpos", " ".join(panda_home_qpos) + " " + fmt_xyz(cube_free_qpos))
    ET.indent(panda_tree, space="  ")
    panda_tree.write(GENERATED_PANDA, encoding="utf-8", xml_declaration=True)

    # 2) Clone the official scene; update the include to use the new Panda.
    scene_tree = ET.parse(SOURCE_SCENE)
    scene_root = scene_tree.getroot()
    include = scene_root.find("./include[@file='panda.xml']")
    if include is None:
        raise ValueError("scene.xml must contain <include file='panda.xml'/>")
    include.set("file", GENERATED_PANDA.name)

    world = scene_root.find("worldbody")
    if world is None:
        raise ValueError("scene.xml is missing <worldbody>")

    # Static collision-enabled tabletop.
    ET.SubElement(world, "geom", {
        "name": "table_top",
        "type": "box",
        "pos": fmt_xyz(TABLE_POS),
        "size": fmt_xyz(TABLE_SIZE),
        "rgba": "0.60 0.43 0.28 1",
        "friction": "1.0 0.01 0.001",
    })

    # A free body has 6 DOF and 7 generalized position values.
    cube_body = ET.SubElement(world, "body", {
        "name": "cube",
        "pos": fmt_xyz(CUBE_POS),
    })
    ET.SubElement(cube_body, "freejoint", {"name": "cube_free"})
    ET.SubElement(cube_body, "geom", {
        "name": "cube_geom",
        "type": "box",
        "size": fmt_xyz((CUBE_HALF_WIDTH,) * 3),
        "mass": str(CUBE_MASS_KG),
        "rgba": "0.17 0.52 0.93 1",
        "friction": "1.2 0.02 0.002",
    })

    # Small marker of where to put the cube in later stages; visual only.
    ET.SubElement(world, "site", {
        "name": "cube_target_marker",
        "pos": fmt_xyz((CUBE_POS[0], CUBE_POS[1], TABLE_POS[2] + TABLE_SIZE[2] + 0.003)),
        "type": "sphere",
        "size": "0.006",
        "rgba": "0.92 0.3 0.12 0.7",
    })

    ET.indent(scene_tree, space="  ")
    scene_tree.write(GENERATED_SCENE, encoding="utf-8", xml_declaration=True)
    print("Created derived files (original XML files are unchanged):")
    print("  ", GENERATED_PANDA)
    print("  ", GENERATED_SCENE)


def object_id(model, kind, name):
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise ValueError(f"Cannot find {name!r} in compiled MuJoCo model")
    return int(idx)


def finger_target(t):
    target = GRIPPER_SCHEDULE[0][1]
    for start_time, value in GRIPPER_SCHEDULE:
        if t >= start_time:
            target = value
    return target


def count_contacts(model, data, cube_geom_id, table_geom_id, finger_bodies):
    """Count cube-table and cube-finger contacts from MuJoCo narrow phase."""
    n_table = 0
    n_finger = 0
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if cube_geom_id not in (a, b):
            continue
        other = b if a == cube_geom_id else a
        if other == table_geom_id:
            n_table += 1
        if int(model.geom_bodyid[other]) in finger_bodies:
            n_finger += 1
    return n_table, n_finger


def main():
    prepare_scene()
    # Avoid native absolute-path problems under Windows folders with Chinese.
    os.chdir(ROOT)
    relative_scene = "mujoco_menagerie/franka_emika_panda/scene_pickplace.xml"
    model = mujoco.MjModel.from_xml_path(relative_scene)
    data = mujoco.MjData(model)

    home_id = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    gripper_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    left_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    right_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    left_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    if model.nu != 8 or gripper_id != 7:
        raise ValueError(
            "Expected official Panda model with 8 actuators and actuator8 at index 7"
        )
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[gripper_id]
    if not (abs(ctrl_lo) < 1e-6 and abs(ctrl_hi - 255) < 1e-6):
        raise ValueError(f"Unexpected gripper ctrlrange: {ctrl_lo}, {ctrl_hi}")

    # All original Panda home values plus cube's free pose are in generated XML.
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    # Hold arm steady at home; do NOT send gripper values to joint1..joint7.
    arm_targets = data.ctrl[:7].copy()
    finger_qpos_adr = [
        int(model.jnt_qposadr[left_id]),
        int(model.jnt_qposadr[right_id]),
    ]
    start_cube_xyz = data.xpos[cube_body].copy()
    dt = float(model.opt.timestep)
    records = []

    print(f"Model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("Gripper actuator: actuator8 (index 7), 0=closed, 255=open")
    print(f"Initial cube XYZ (m): {start_cube_xyz}")
    print("Stage 1: gripper moves above the table; it does not grasp the cube yet.")

    if SHOW_VIEWER:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    else:
        from contextlib import nullcontext
        viewer_ctx = nullcontext(None)

    with viewer_ctx as viewer:
        while data.time < TOTAL_TIME:
            if viewer is not None and not viewer.is_running():
                break
            wall = time.perf_counter()
            current_command = finger_target(data.time)
            data.ctrl[:7] = arm_targets
            data.ctrl[gripper_id] = current_command
            mujoco.mj_step(model, data)

            left_m = float(data.qpos[finger_qpos_adr[0]])
            right_m = float(data.qpos[finger_qpos_adr[1]])
            opening_mm = 1000 * (left_m + right_m)
            cube_xyz = data.xpos[cube_body].copy()
            table_n, finger_n = count_contacts(
                model, data, cube_geom, table_geom, {left_body, right_body}
            )
            records.append([
                float(data.time),
                float(current_command),
                left_m, right_m, opening_mm,
                *[float(v) for v in cube_xyz],
                table_n, finger_n,
            ])

            if viewer is not None:
                viewer.sync()
                if REALTIME_WHEN_VISIBLE:
                    sleep_seconds = dt - (time.perf_counter() - wall)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

    if not records:
        raise RuntimeError("No simulation steps recorded")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "gripper_log.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "gripper_command_0_255",
            "left_finger_m", "right_finger_m", "opening_mm",
            "cube_x_m", "cube_y_m", "cube_z_m",
            "cube_table_contacts", "cube_finger_contacts",
        ])
        writer.writerows(records)

    arr = np.asarray(records)
    t = arr[:, 0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.step(t, arr[:, 1], where="post", label="Command (0=close, 255=open)")
    ax1.set_ylabel("Command (0-255)")
    ax1.set_ylim(-15, 270)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(t, arr[:, 4], label="Actual finger opening")
    ax2.set_ylabel("Opening width (mm)")
    ax2.set_xlabel("Simulation time (s)")
    ax2.set_ylim(-3, 88)
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    plot_path = RESULTS / "gripper_open_close.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Final cube XYZ (m): {arr[-1, 5:8]}")
    print(f"Final finger opening (mm): {arr[-1, 4]:.2f}")
    print(f"Max cube-table contacts in one step: {int(arr[:, 8].max())}")
    print(f"Max cube-finger contacts in one step: {int(arr[:, 9].max())}")
    print("CSV:", path)
    print("Plot:", plot_path)
    print("If no cube-finger contact is reported, that is EXPECTED in Stage 1.")


if __name__ == "__main__":
    main()
