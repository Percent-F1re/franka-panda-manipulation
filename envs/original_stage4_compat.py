"""Unmodified numerical constants / MJCF index conventions from original Stage1-4.

NO import of Stage4: its module top level compiles a separate environment and
changes cwd. Values below are copied verbatim from the uploaded sources.
"""
from __future__ import annotations
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _required_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise RuntimeError(f'Original model is missing MJCF {kind} {name!r}')
    return int(index)


def count_cube_contacts(model, data, cube_geom, table_geom, left_body, right_body):
    # Exact original Stage2 function, counts cube/table and cube/finger contacts.
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


def orientation_error_world(R_target, R_actual):
    return Rotation.from_matrix(R_target @ R_actual.T).as_rotvec()


def make_original_stage4_adapter(model):
    if model.nu != 8:
        raise RuntimeError(f'Original Panda must have 8 actuators; got {model.nu}')
    for i in range(8):
        actual_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actual_name != f'actuator{i+1}':
            raise RuntimeError(f'Actuator ordering changed at {i}: {actual_name}')
    home = _required_id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    arm_joint_ids = np.asarray([
        _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
        for i in range(1, 8)
    ], dtype=int)
    qpos_ids = model.jnt_qposadr[arm_joint_ids]
    dof_ids = model.jnt_dofadr[arm_joint_ids]
    # Original Stage4 qlo/qhi intersection + 0.02 rad margin.
    qlo = np.maximum(model.jnt_range[arm_joint_ids, 0], model.actuator_ctrlrange[:7, 0]) + .02
    qhi = np.minimum(model.jnt_range[arm_joint_ids, 1], model.actuator_ctrlrange[:7, 1]) - .02
    if np.any(qlo >= qhi):
        raise RuntimeError('Invalid original Panda joint limits')
    cube_joint = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free')
    finger_joints = [
        _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ('finger_joint1', 'finger_joint2')
    ]
    s3 = SimpleNamespace(
        SETTLE_BEFORE_GRASP=1.5,
        MIN_LIFT_MM=40.0,
        XY_TOLERANCE_MM=20.0,
        Z_TOLERANCE_MM=10.0,
        STABLE_SPEED_M_S=.03,
        STABLE_DRIFT_MM=10.0,
    )
    return SimpleNamespace(
        model=model,
        s3=s3,
        home=home,
        hand=_required_id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand'),
        cube=_required_id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube'),
        cube_geom=_required_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom'),
        table_geom=_required_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top'),
        left=_required_id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger'),
        right=_required_id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger'),
        qpos_ids=qpos_ids,
        dof_ids=dof_ids,
        cube_qpos_addr=int(model.jnt_qposadr[cube_joint]),
        cube_dof_addr=int(model.jnt_dofadr[cube_joint]),
        finger_qpos_ids=[int(model.jnt_qposadr[j]) for j in finger_joints],
        grip_act=_required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8'),
        qlo=qlo.copy(), qhi=qhi.copy(),
        CUBE_POS=np.array([.58, 0, .44], dtype=float),
        CUBE_HALF_WIDTH=.020,
        HAND_TO_GRASP_LOCAL=np.array([0, 0, .1029], dtype=float),
        GRIPPER_OPEN=255.0,
        GRIPPER_CLOSED=0.0,
        TRANSFER_XY=np.array([0.0, .06], dtype=float),
        K_POS=2.5,
        K_ROT=2.0,
        DAMPING=.05,
        MAX_JOINT_CMD_SPEED=.3,
        orientation_error_world=orientation_error_world,
        count_cube_contacts=count_cube_contacts,
    )
