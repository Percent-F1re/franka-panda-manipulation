"""Gymnasium Franka Panda 3D reaching environment (joint-position delta actions).

Place this file under robot_simulation/sac_reach/ ; the official MuJoCo
Menagerie model should remain in robot_simulation/mujoco_menagerie/.
This is NOT a real-robot controller. V2 ablation: add accumulated commanded joint positions to the observation.
Do not load V1 SAC weights with this 30-D V2 environment.
"""
from pathlib import Path
import os
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class PandaReachEnvV2(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(
        self,
        model_path=None,
        render_mode=None,
        goal_range=0.08,
        max_episode_steps=125,
        frame_skip=20,
        action_delta_rad=0.04,
        distance_threshold=0.005,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self._viewer = None
        self._project_root = Path(__file__).resolve().parents[2]  # script lives in robot_simulation/sac_reach/
        model_path = Path(model_path) if model_path is not None else (
            Path("mujoco_menagerie") / "franka_emika_panda" / "scene.xml"
        )
        if not model_path.is_absolute():
            model_path = self._project_root / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Cannot locate XML: {model_path}")
        # On some Windows setups the MuJoCo XML parser has trouble with Chinese
        # absolute paths. Give it a relative ASCII path from project root.
        # This assumes project-relative scene.xml and referenced assets.
        if model_path.is_relative_to(self._project_root):
            relative_path = model_path.relative_to(self._project_root)
            old_dir = Path.cwd()
            try:
                os.chdir(self._project_root)
                self.model = mujoco.MjModel.from_xml_path(relative_path.as_posix())
            finally:
                os.chdir(old_dir)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        def named_id(obj_type, name):
            idx = mujoco.mj_name2id(self.model, obj_type, name)
            if idx < 0:
                raise ValueError(f"Name '{name}' not found in Panda model")
            return idx

        self.home_id = named_id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self.hand_id = named_id(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.joint_ids = np.array([
            named_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ], dtype=int)
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        if self.model.nu < 8:
            raise ValueError("Expected 7 arm actuators plus Panda gripper actuator")
        if not np.all(self.model.jnt_limited[self.joint_ids]):
            raise ValueError("All arm joints require joint limits")
        if not np.all(self.model.actuator_ctrllimited[:7]):
            raise ValueError("All arm actuators require control limits")
        self.lower = np.maximum(
            self.model.jnt_range[self.joint_ids, 0],
            self.model.actuator_ctrlrange[:7, 0],
        ) + 0.02
        self.upper = np.minimum(
            self.model.jnt_range[self.joint_ids, 1],
            self.model.actuator_ctrlrange[:7, 1],
        ) - 0.02
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid actuator joint limit overlap")

        self.goal_range = float(goal_range)
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.action_delta_rad = float(action_delta_rad)
        self.distance_threshold = float(distance_threshold)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        # qpos(7), qvel(7), current_xyz(3), goal_xyz(3), delta_xyz(3), q_command(7)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )
        self.step_count = 0
        self.success_streak = 0
        self.goal = np.zeros(3, dtype=float)
        self._home_xyz = None
        self._q_command = np.zeros(7, dtype=float)
        self._initialize_home()

    def _initialize_home(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        self.data.qvel[:] = 0
        self.data.qacc_warmstart[:] = 0
        self._q_command = np.clip(
            self.data.qpos[self.qpos_ids].copy(), self.lower, self.upper
        )
        self.data.ctrl[:7] = self._q_command
        self.data.ctrl[7] = 255.0  # gripper open, not used for Reach
        mujoco.mj_forward(self.model, self.data)
        self._home_xyz = self.data.xpos[self.hand_id].copy()

    def _get_obs(self):
        xyz = self.data.xpos[self.hand_id].copy()
        return np.concatenate((
            self.data.qpos[self.qpos_ids],
            self.data.qvel[self.qvel_ids],
            xyz,
            self.goal,
            self.goal - xyz,
            self._q_command.copy(),
        )).astype(np.float32)

    def _distance(self):
        return float(np.linalg.norm(self.goal - self.data.xpos[self.hand_id]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_home()
        self.step_count = 0
        self.success_streak = 0
        if options is not None and "goal" in options:
            goal = np.asarray(options["goal"], dtype=float)
            if goal.shape != (3,):
                raise ValueError("options['goal'] must be XYZ shape (3,)")
            self.goal = goal.copy()
        else:
            self.goal = self._home_xyz + self.np_random.uniform(
                -self.goal_range, self.goal_range, size=3
            )
        distance = self._distance()
        return self._get_obs(), {"distance_mm": distance * 1000, "is_success": False}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite vector of shape (7,)")
        action = np.clip(action, -1.0, 1.0)
        old_dist = self._distance()
        self._q_command = np.clip(
            self._q_command + action * self.action_delta_rad,
            self.lower,
            self.upper,
        )
        self.data.ctrl[:7] = self._q_command
        self.data.ctrl[7] = 255.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        new_dist = self._distance()
        inside = new_dist <= self.distance_threshold
        self.success_streak = self.success_streak + 1 if inside else 0
        terminated = self.success_streak >= 3  # remain in target for 3 steps
        truncated = self.step_count >= self.max_episode_steps and not terminated
        # Dense reward: make progress and stay near goal. Bonus for stable arrival.
        # Units deliberately scaled by 0.1 m for human-readable rewards.
        reward = 5.0 * (old_dist - new_dist) / 0.1 - new_dist / 0.1
        reward -= 0.005 * float(np.dot(action, action))
        if terminated:
            reward += 10.0
        info = {
            "distance_mm": new_dist * 1000.0,
            "is_success": bool(terminated),
            "step_count": self.step_count,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "human":
            return None
        import mujoco.viewer
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer.is_running():
            self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
