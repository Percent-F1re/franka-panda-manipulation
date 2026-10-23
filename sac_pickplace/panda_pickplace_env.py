"""Gymnasium environment for learning physical Franka Panda grasp-and-place.

Built on the user's *working* Stage-4 MuJoCo scene; unlike Stage 4, this
module does NOT execute any scripted approach/grasp/lift/transfer/place stages.
A policy controls Cartesian XYZ increments and the gripper; a 6D DLS
controller converts Cartesian targets into joint position commands.

INSTALL LOCATION: robot_simulation/sac_pickplace/panda_pickplace_env.py
REQUIRES: the user's panda_pickplace_stage1/2_grasp/3/4_random.py in
          robot_simulation/ and the matching mujoco_menagerie directory.

First release: development/smoke-test environment, NOT a verified learning
benchmark. Run smoke_test_pickplace_env.py before starting training.
"""
from __future__ import annotations

import importlib
import sys
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


def _load_stage4():
    """Find original Stage-4 files without hard-coding a Windows path."""
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    project_root = next(
        (p for p in candidates if (p / 'panda_pickplace_stage4_random.py').is_file()),
        None,
    )
    if project_root is None:
        raise FileNotFoundError(
            'Cannot locate panda_pickplace_stage4_random.py. Put this file in '
            'robot_simulation/sac_pickplace/ and keep the Stage 1-4 scripts '
            'in robot_simulation/.'
        )
    missing = [name for name in (
        'panda_pickplace_stage1.py', 'panda_pickplace_stage2_grasp.py',
        'panda_pickplace_stage3.py') if not (project_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Stage-4 dependencies in {project_root}: {missing}')
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Stage4 builds a model at import and may change CWD. This is retained to
    # preserve its exact, known-working scene generation and geometry setup.
    return importlib.import_module('panda_pickplace_stage4_random')


class PandaPickPlaceEnv(gym.Env):
    """State-based SAC environment using a position-controlled Panda.

    action: float32[4], each in [-1,1]
      action[:3]: delta hand XYZ (max_xyz_step_m per policy step)
      action[3]: gripper setpoint (positive=open, negative=close)

    task='lift' (recommended first): approach, grasp, physically lift cube.
    task='pickplace': physically lift, transfer, release and settle at target.

    Observations intentionally contain *privileged simulated state* (including
    cube world position); this is NOT a vision-based or sim-to-real policy.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 25}

    def __init__(
        self,
        task: str = 'lift',
        render_mode: str | None = None,
        random_xy_range_m: float = 0.04,
        max_xyz_step_m: float = 0.01,
        control_period_s: float = 0.04,
        max_episode_steps: int = 250,
        hold_lift_steps: int = 5,
        place_stable_seconds: float = 0.6,
    ):
        super().__init__()
        if task not in ('lift', 'pickplace'):
            raise ValueError("task must be 'lift' or 'pickplace'")
        if render_mode not in (None, 'human'):
            raise ValueError("render_mode must be None or 'human'")
        if random_xy_range_m < 0 or max_xyz_step_m <= 0 or control_period_s <= 0:
            raise ValueError('Invalid step/range parameter')
        if max_episode_steps < 1 or hold_lift_steps < 1:
            raise ValueError('Invalid episode length / hold_lift_steps')

        self.s4 = _load_stage4()
        self.s3 = self.s4.s3
        # Stage4's model is compiled once, but each Env gets its OWN MjData.
        # For now use one environment instance per process while prototyping.
        self.model = self.s4.model
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.render_mode = render_mode
        self.viewer = None
        self.dt = float(self.model.opt.timestep)
        self.frame_skip = max(1, round(control_period_s / self.dt))
        self.control_dt = self.frame_skip * self.dt
        self.max_xyz_step_m = float(max_xyz_step_m)
        self.random_xy_range_m = float(random_xy_range_m)
        self.max_episode_steps = int(max_episode_steps)
        self.hold_lift_steps = int(hold_lift_steps)
        self.place_window_steps = max(2, int(np.ceil(place_stable_seconds / self.control_dt)))

        # Reuse the EXACT Panda geometry, indices and limits from Stage4.
        self.home = self.s4.home
        self.hand = self.s4.hand
        self.cube = self.s4.cube
        self.cube_geom = self.s4.cube_geom
        self.table_geom = self.s4.table_geom
        self.left = self.s4.left
        self.right = self.s4.right
        self.qpos_ids = self.s4.qpos_ids.copy()
        self.dof_ids = self.s4.dof_ids.copy()
        self.cube_qpos_addr = self.s4.cube_qpos_addr
        self.cube_dof_addr = self.s4.cube_dof_addr
        self.finger_qpos_ids = list(self.s4.finger_qpos_ids)
        self.grip_act = self.s4.grip_act
        self.qlo = self.s4.qlo.copy()
        self.qhi = self.s4.qhi.copy()
        self.jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.gripper_open = float(self.s4.GRIPPER_OPEN)
        self.gripper_closed = float(self.s4.GRIPPER_CLOSED)
        self.transfer_xy = np.asarray(self.s4.TRANSFER_XY, dtype=float).copy()
        self.grasp_local = np.asarray(self.s4.HAND_TO_GRASP_LOCAL, dtype=float).copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # Set after reset because we generate a dimension-consistent reference obs.
        self.step_count = 0
        self.q_cmd = np.zeros(7, dtype=float)
        self.gripper_target = self.gripper_open
        self.hand_target = np.zeros(3, dtype=float)
        self.R_target = np.eye(3, dtype=float)
        self.cube_start = np.zeros(3, dtype=float)
        self.target_xy = np.zeros(2, dtype=float)
        self.table_z = 0.0
        self.has_lifted = False
        self.has_transferred = False
        self.lift_streak = 0
        self.place_streak = 0
        self.place_history = deque(maxlen=self.place_window_steps)
        self.prev_grasp_distance = 0.0
        self.prev_cube_target_distance = 0.0
        self.prev_cube_rise = 0.0
        self.clip_steps = 0
        self.workspace_low = np.zeros(3, dtype=float)
        self.workspace_high = np.zeros(3, dtype=float)

        obs, _ = self.reset(seed=0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _pose(self):
        return (
            self.data.xpos[self.hand].copy(),
            self.data.xmat[self.hand].reshape(3, 3).copy(),
        )

    def _contacts(self) -> tuple[int, int, int]:
        return self.s4.count_cube_contacts(
            self.model, self.data,
            self.cube_geom, self.table_geom, self.left, self.right,
        )

    def _cube_velocity(self) -> np.ndarray:
        return self.data.qvel[self.cube_dof_addr:self.cube_dof_addr+3].copy()

    def _grasp_distance(self) -> float:
        hand_pos, R = self._pose()
        grasp_point = hand_pos + R @ self.grasp_local
        return float(np.linalg.norm(grasp_point - self.data.xpos[self.cube]))

    def _get_obs(self) -> np.ndarray:
        hand_p, R = self._pose()
        mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.hand)
        hand_vel = self.jacp[:, self.dof_ids] @ self.data.qvel[self.dof_ids]
        cube_p = self.data.xpos[self.cube].copy()
        tbl, lc, rc = self._contacts()
        opening = sum(float(self.data.qpos[i]) for i in self.finger_qpos_ids)
        denom = self.gripper_open - self.gripper_closed
        grip_fraction = (self.gripper_target - self.gripper_closed) / denom if abs(denom) > 1e-9 else 0.0
        obs = np.concatenate([
            self.data.qpos[self.qpos_ids],              # 7
            self.data.qvel[self.dof_ids],               # 7
            self.q_cmd,                                 # 7
            hand_p,                                     # 3
            hand_vel,                                   # 3
            cube_p,                                     # 3
            self._cube_velocity(),                      # 3
            cube_p - hand_p,                            # 3
            self.target_xy - cube_p[:2],               # 2
            np.array([opening, grip_fraction]),         # 2
            np.array([float(self.has_lifted), float(self.has_transferred)]),  # 2
            np.array([self.step_count / self.max_episode_steps]),             # 1
            np.array([float(tbl > 0), float(lc > 0), float(rc > 0)]),         # 3
        ]).astype(np.float32)
        return obs  # 46 dimensions

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        # Testing can reproduce *exactly the same target offset* as Stage 4.
        if 'xy_offset' in options:
            offset = np.asarray(options['xy_offset'], dtype=float)
        else:
            offset = self.np_random.uniform(
                -self.random_xy_range_m, self.random_xy_range_m, size=2
            )
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("options['xy_offset'] must be two finite numbers")
        if np.any(np.abs(offset) > self.random_xy_range_m + 1e-10):
            raise ValueError('xy_offset exceeds configured random_xy_range_m')

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home)
        start_xy = np.asarray(self.s4.CUBE_POS[:2], dtype=float) + offset
        self.data.qpos[self.cube_qpos_addr:self.cube_qpos_addr+3] = [
            start_xy[0], start_xy[1], float(self.s4.CUBE_POS[2])
        ]
        self.data.qpos[self.cube_qpos_addr+3:self.cube_qpos_addr+7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0
        self.data.qacc_warmstart[:] = 0
        self.data.ctrl[:] = 0
        self.q_cmd = np.clip(self.data.qpos[self.qpos_ids].copy(), self.qlo, self.qhi)
        self.data.ctrl[:7] = self.q_cmd
        self.gripper_target = self.gripper_open
        self.data.ctrl[self.grip_act] = self.gripper_target
        mujoco.mj_forward(self.model, self.data)
        # Same physically settled cube initialization as the Stage-4 expert.
        for _ in range(max(1, round(float(self.s3.SETTLE_BEFORE_GRASP) / self.dt))):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.cube_start = self.data.xpos[self.cube].copy()
        self.target_xy = self.cube_start[:2] + self.transfer_xy
        self.table_z = float(
            self.data.geom_xpos[self.table_geom, 2]
            + self.model.geom_size[self.table_geom, 2]
        )
        self.hand_target, self.R_target = self._pose()
        alignment = float(self.R_target[:, 2] @ np.array([0, 0, -1.0]))
        if alignment < 0.90:
            raise RuntimeError(f'Gripper not pointing downward; alignment={alignment:.3f}')
        self.workspace_low = np.array([
            self.hand_target[0] - 0.25,
            self.hand_target[1] - 0.25,
            max(self.hand_target[2] - 0.25, self.table_z + 0.03),
        ])
        self.workspace_high = np.array([
            self.hand_target[0] + 0.25,
            self.hand_target[1] + 0.25,
            self.hand_target[2] + 0.20,
        ])
        self.step_count = 0
        self.has_lifted = False
        self.has_transferred = False
        self.lift_streak = 0
        self.place_streak = 0
        self.place_history.clear()
        self.prev_grasp_distance = self._grasp_distance()
        self.prev_cube_target_distance = float(
            np.linalg.norm(self.data.xpos[self.cube, :2] - self.target_xy)
        )
        self.prev_cube_rise = 0.0
        self.clip_steps = 0
        return self._get_obs(), self._info(False, False)

    def _apply_dls(self):
        hand_p, R = self._pose()
        pos_error = self.hand_target - hand_p
        ori_error = self.s4.orientation_error_world(self.R_target, R)
        mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.hand)
        J = np.vstack((self.jacp[:, self.dof_ids], self.jacr[:, self.dof_ids]))
        twist = np.concatenate((self.s4.K_POS * pos_error, self.s4.K_ROT * ori_error))
        damping = float(self.s4.DAMPING)
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), twist)
        peak = np.max(np.abs(dq))
        max_speed = float(self.s4.MAX_JOINT_CMD_SPEED)
        if peak > max_speed:
            dq *= max_speed / peak
        new_q = self.q_cmd + dq * self.dt
        self.clip_steps += int(np.any((new_q < self.qlo) | (new_q > self.qhi)))
        self.q_cmd = np.clip(new_q, self.qlo, self.qhi)
        self.data.ctrl[:7] = self.q_cmd
        self.data.ctrl[self.grip_act] = self.gripper_target
        mujoco.mj_step(self.model, self.data)

    def _physical_state(self):
        cube_p = self.data.xpos[self.cube].copy()
        tbl, lc, rc = self._contacts()
        rise = float(cube_p[2] - self.cube_start[2])
        grasped = bool(lc > 0 and rc > 0 and tbl == 0)
        lifted = bool(grasped and rise >= self.s3.MIN_LIFT_MM / 1000.0)
        moved = float(np.linalg.norm(cube_p[:2] - self.cube_start[:2]))
        transferred = bool(
            self.has_lifted and tbl == 0
            and rise >= self.s3.MIN_LIFT_MM / 1000.0
            and moved >= 0.035
        )
        bottom_gap = float(cube_p[2] - self.s4.CUBE_HALF_WIDTH - self.table_z)
        xy_error = float(np.linalg.norm(cube_p[:2] - self.target_xy))
        speed = float(np.linalg.norm(self._cube_velocity()))
        return {
            'cube_xyz_m': cube_p,
            'table_contact': int(tbl), 'left_contact': int(lc), 'right_contact': int(rc),
            'grasped': grasped, 'currently_lifted': lifted,
            'currently_transferred': transferred,
            'rise_m': rise, 'cube_xy_error_m': xy_error,
            'bottom_gap_m': bottom_gap, 'cube_speed_m_s': speed,
        }

    def _info(self, success: bool, truncated: bool) -> dict[str, Any]:
        s = self._physical_state()
        return {
            'is_success': bool(success), 'task': self.task,
            'steps': int(self.step_count), 'sim_time_s': float(self.step_count * self.control_dt),
            'cube_xyz_m': s['cube_xyz_m'].tolist(),
            'target_xy_m': self.target_xy.tolist(),
            'grasped': s['grasped'], 'has_lifted': bool(self.has_lifted),
            'has_transferred': bool(self.has_transferred),
            'cube_xy_error_mm': s['cube_xy_error_m'] * 1000,
            'cube_rise_mm': s['rise_m'] * 1000,
            'cube_speed_m_s': s['cube_speed_m_s'],
            'contacts': (s['table_contact'], s['left_contact'], s['right_contact']),
            'place_streak': int(self.place_streak),
            'joint_clip_steps': int(self.clip_steps),
            'time_limit_reached': bool(truncated),
        }

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError('Action must be a finite vector of shape (4,)')
        a = np.clip(a, -1.0, 1.0)
        # Policy decides the displacement; DLS only solves inverse kinematics.
        self.hand_target = np.clip(
            self.hand_target + self.max_xyz_step_m * a[:3],
            self.workspace_low, self.workspace_high,
        )
        # Gripper command is an absolute target, not an expert stage switch.
        self.gripper_target = (
            0.5 * (float(a[3]) + 1) * self.gripper_open
            + 0.5 * (1 - float(a[3])) * self.gripper_closed
        )
        for _ in range(self.frame_skip):
            self._apply_dls()
        self.step_count += 1
        state = self._physical_state()
        grasp_distance = self._grasp_distance()
        cube_target_dist = state['cube_xy_error_m']
        cube_rise = state['rise_m']

        # Progress-shaped reward; event bonuses are paid exactly once.
        progress = float(np.clip(self.prev_grasp_distance - grasp_distance, -0.03, 0.03))
        reward = -0.01 + 15.0 * progress - 0.005 * float(np.sum(a[:3]**2))
        if state['grasped']:
            reward += 0.04
        if state['currently_lifted']:
            self.lift_streak += 1
        else:
            self.lift_streak = 0
        if self.lift_streak >= self.hold_lift_steps and not self.has_lifted:
            self.has_lifted = True
            reward += 10.0
        if self.has_lifted and not self.has_transferred:
            # Reward actual physical cube motion only while it's raised.
            if state['table_contact'] == 0 and cube_rise >= self.s3.MIN_LIFT_MM / 1000.0:
                reward += 12.0 * float(np.clip(self.prev_cube_target_distance - cube_target_dist, -0.03, 0.03))
            if state['currently_transferred']:
                self.has_transferred = True
                reward += 12.0
        if self.has_lifted:
            reward += 2.0 * float(np.clip(cube_rise - self.prev_cube_rise, -0.03, 0.03))

        # Every successful-place frame MUST satisfy real contact/release tests.
        # 0.6s history prevents awarding a quick touch-and-go 'placement'.
        self.place_history.append(state['cube_xyz_m'].copy())
        history = np.asarray(self.place_history)
        stable_drift_m = (
            float(np.max(np.linalg.norm(history - history[0], axis=1)))
            if len(history) >= self.place_window_steps else float('inf')
        )
        released = (state['table_contact'] > 0 and
                    state['left_contact'] == 0 and state['right_contact'] == 0)
        placed_now = bool(
            self.has_lifted and self.has_transferred and released
            and cube_target_dist * 1000 <= self.s3.XY_TOLERANCE_MM
            and abs(state['bottom_gap_m']) * 1000 <= self.s3.Z_TOLERANCE_MM
            and state['cube_speed_m_s'] <= self.s3.STABLE_SPEED_M_S
            and stable_drift_m * 1000 <= self.s3.STABLE_DRIFT_MM
        )
        self.place_streak = self.place_streak + 1 if placed_now else 0
        success = bool(self.has_lifted) if self.task == 'lift' else bool(self.place_streak >= 3)
        if success:
            reward += 30.0 if self.task == 'lift' else 50.0
        truncated = bool(self.step_count >= self.max_episode_steps and not success)
        self.prev_grasp_distance = grasp_distance
        self.prev_cube_target_distance = cube_target_dist
        self.prev_cube_rise = cube_rise
        obs = self._get_obs()
        info = self._info(success, truncated)
        info['stable_drift_mm'] = stable_drift_m * 1000
        info['grasp_distance_mm'] = grasp_distance * 1000
        return obs, float(reward), success, truncated, info

    def render(self):
        if self.render_mode == 'human':
            import mujoco.viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = PandaPickPlaceEnv(task='lift')
    obs, info = env.reset(seed=42)
    print('Observation shape:', obs.shape)
    print('Action space:', env.action_space)
    print('Reset info:', info)
    for i in range(10):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        print(f'{i+1}: reward={rew:.3f}, grasped={info["grasped"]}, rise={info["cube_rise_mm"]:.1f}mm')
        if term or trunc:
            break
    env.close()
