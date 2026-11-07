"""Selectable controller 2: *actual robosuite 1.5.2* OSC_POSE.

Robosuite OSC returns torques; original Menagerie actuators 1..7 are
position servos. Only in OSC mode, convert these compiled general actuators
into direct-torque motors. Their gear, original force limits, robot mechanics,
gripper actuator8, original cube and table remain unchanged.

The old SAC policy has 46D input including 7D q_cmd. For OSC mode, compute a
SHADOW q_cmd using the original DLS method solely to preserve the observation
interface. Shadow q_cmd is NOT applied as the arm's real control signal.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from envs.dual_scene_core import EEF_SITE, CENTER_SITE


class RobosuiteOSCBridge:
    name = 'osc'
    observation_q_cmd_semantics = 'virtual DLS reference only, not executed torque'

    def __init__(self, task_env, original_dls_substep_fn, *, kp=150.0, damping_ratio=1.0):
        if float(kp) <= 0 or float(damping_ratio) <= 0:
            raise ValueError('OSC kp and damping_ratio must be positive')
        self.env = task_env
        self.original_dls_substep_fn = original_dls_substep_fn
        self.kp = float(kp)
        self.damping_ratio = float(damping_ratio)
        self.model = task_env.model
        self.osc = None
        self.torque_enabled = False
        self.last_goal_step = None
        self.substeps = 0
        self.saturated_substeps = 0
        self.peak_unclipped_tau = 0.0
        self._saved = {
            key: np.array(getattr(self.model, key)[:7], copy=True)
            for key in ('actuator_gaintype', 'actuator_biastype',
                        'actuator_gainprm', 'actuator_biasprm',
                        'actuator_ctrlrange', 'actuator_ctrllimited')
        }
        self.tau_low, self.tau_high = self._check_original_actuators()

    def _check_original_actuators(self):
        model = self.model
        if model.nu != 8:
            raise RuntimeError(f'Original Menagerie Panda needs 8 actuators; got {model.nu}')
        for i in range(7):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name != f'actuator{i+1}':
                raise RuntimeError(f'Original actuator ordering changed: #{i} is {name!r}')
            # Torque conversion is correct ONLY for a one-to-one unit-gear
            # transmission from motor to this joint.
            gear = np.asarray(model.actuator_gear[i], dtype=float)
            if abs(gear[0] - 1.0) > 1e-9 or not np.allclose(gear[1:], 0.0, atol=1e-9):
                raise RuntimeError(f'Unexpected non-unit joint gear on actuator{i+1}: {gear}')
            if int(model.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise RuntimeError(f'Actuator{i+1} must have a direct joint transmission')
            if not bool(model.actuator_forcelimited[i]):
                raise RuntimeError(f'Original actuator{i+1} has no force limit; cannot compare fairly')
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE) < 0:
            raise RuntimeError('Required dual OSC hand-body reference site is missing')
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CENTER_SITE) < 0:
            raise RuntimeError('Required static OSC center site is missing')
        force = np.asarray(model.actuator_forcerange[:7], dtype=float)
        if force.shape != (7, 2) or not np.all(np.isfinite(force)) or np.any(force[:, 0] >= force[:, 1]):
            raise RuntimeError('Original joint force limits cannot be used as OSC torque limits')
        return force[:, 0].copy(), force[:, 1].copy()

    def enter_torque_mode(self):
        """Convert only 7 original *compiled* arm actuator semantics."""
        if self.torque_enabled:
            return
        m = self.model
        m.actuator_gaintype[:7] = int(mujoco.mjtGain.mjGAIN_FIXED)
        m.actuator_biastype[:7] = int(mujoco.mjtBias.mjBIAS_NONE)
        m.actuator_gainprm[:7] = 0.0
        m.actuator_gainprm[:7, 0] = 1.0
        m.actuator_biasprm[:7] = 0.0
        m.actuator_ctrlrange[:7, 0] = self.tau_low
        m.actuator_ctrlrange[:7, 1] = self.tau_high
        m.actuator_ctrllimited[:7] = True
        self.torque_enabled = True
        # Reinterpretation of the old q-command as a torque is unsafe even
        # during mj_forward; clear arm controls before updating model state.
        self.env.data.ctrl[:7] = 0.0
        mujoco.mj_forward(m, self.env.data)

    def exit_torque_mode(self):
        """Restore EXACT original actuator semantics before the legacy reset."""
        if not self.torque_enabled:
            return
        for key, saved in self._saved.items():
            getattr(self.model, key)[:7] = saved
        self.torque_enabled = False
        self.osc = None
        self.last_goal_step = None
        self.env.data.ctrl[:7] = np.clip(
            self.env.data.qpos[self.env.qpos_ids], self.env.qlo, self.env.qhi
        )
        mujoco.mj_forward(self.model, self.env.data)

    def reset_after_original_settle(self):
        """Call after parent's reset using original position servo and settling."""
        self.enter_torque_mode()
        # Official robosuite OSC_POSE, not a homebrew OSC approximation.
        from robosuite.controllers.parts.arm.osc import OperationalSpaceController

        e = self.env
        self.osc = OperationalSpaceController(
            sim=e.raw_env.sim,
            ref_name=EEF_SITE,
            joint_indexes={
                'joints': [
                    int(mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}'))
                    for i in range(1, 8)
                ],
                'qpos': e.qpos_ids.tolist(),
                'qvel': e.dof_ids.tolist(),
            },
            actuator_range=(self.tau_low.copy(), self.tau_high.copy()),
            kp=self.kp,
            damping_ratio=self.damping_ratio,
            impedance_mode='fixed',
            policy_freq=int(round(1.0 / e.control_dt)),
            control_ori=True,
            input_type='absolute',
            input_ref_frame='world',
            uncouple_pos_ori=True,
            interpolator_pos=None,
            interpolator_ori=None,
            # robosuite OSC 1.5.2 reads hand_center as world-fixed base site
            part_name='hand',
            naming_prefix='',
            lite_physics=True,
        )
        self.osc.update_initial_joints(e.data.qpos[e.qpos_ids].copy())
        self.osc.reset_goal(goal_update_mode='desired')
        self.last_goal_step = None
        self.substeps = 0
        self.saturated_substeps = 0
        self.peak_unclipped_tau = 0.0
        # Arm commands must now be torques, NEVER the keyframe's q positions.
        e.data.ctrl[:7] = 0.0
        e.data.ctrl[e.grip_act] = e.gripper_target

    def _set_goal_if_new_policy_step(self):
        e = self.env
        if self.last_goal_step == e.step_count:
            return
        pose6 = np.concatenate((
            np.asarray(e.hand_target, dtype=float),
            Rotation.from_matrix(e.R_target).as_rotvec(),
        ))
        if pose6.shape != (6,) or not np.all(np.isfinite(pose6)):
            raise FloatingPointError('Invalid 6D absolute pose fed into robosuite OSC')
        # input_type='absolute' bypasses OSC action scaling. The target is the
        # EXACT legacy workspace-clipped XYZ and original fixed 3D orientation.
        self.osc.set_goal(pose6)
        self.last_goal_step = int(e.step_count)

    def run_substep(self):
        if not self.torque_enabled or self.osc is None:
            raise RuntimeError('OSC substep called before torque-mode reset')
        # Exact legacy DLS reference for q_cmd observation ONLY; the motor
        # command it writes is immediately REPLACED with OSC torques below.
        prior_arm_ctrl = self.env.data.ctrl[:7].copy()
        try:
            self.original_dls_substep_fn()
        finally:
            # q_cmd is updated for observation compatibility but its position
            # servo signal must NEVER be used in OSC torque mode.
            self.env.data.ctrl[:7] = prior_arm_ctrl
        self._set_goal_if_new_policy_step()
        self._apply_osc_torques()

    def hold_goal_substep(self):
        """No SAC, no DLS shadow update; continue OSC tracking frozen pose."""
        if not self.torque_enabled or self.osc is None:
            raise RuntimeError('Cannot hold OSC goal before controller initialization')
        self._apply_osc_torques()

    def _apply_osc_torques(self):
        torques = np.asarray(self.osc.run_controller(), dtype=float).reshape(-1)
        if torques.shape != (7,) or not np.all(np.isfinite(torques)):
            raise FloatingPointError(f'Official robosuite OSC produced invalid torques: {torques}')
        self.peak_unclipped_tau = max(self.peak_unclipped_tau, float(np.max(np.abs(torques))))
        if np.any((torques < self.tau_low) | (torques > self.tau_high)):
            self.saturated_substeps += 1
        self.substeps += 1
        self.env.data.ctrl[:7] = np.clip(torques, self.tau_low, self.tau_high)
        self.env.data.ctrl[self.env.grip_act] = self.env.gripper_target

    def diagnostics(self):
        return {
            'controller': self.name,
            'q_cmd_semantics': self.observation_q_cmd_semantics,
            'osc_kp': self.kp,
            'osc_damping_ratio': self.damping_ratio,
            'torque_saturated_substeps': self.saturated_substeps,
            'torque_substeps': self.substeps,
            'torque_saturation_fraction': (
                self.saturated_substeps / self.substeps if self.substeps else 0.0
            ),
            'peak_unclipped_torque_nm': self.peak_unclipped_tau,
        }
