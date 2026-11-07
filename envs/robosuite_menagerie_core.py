"""robosuite 1.5.2 *MujocoEnv* with the ORIGINAL Menagerie Panda MJCF.

Crucial architectural choice: we do not instantiate robosuite's different Panda
robot, gripper, or JOINT_POSITION impedance controller.  Instead, robosuite
MujocoEnv/MjSim own and advance the MuJoCo model compiled from the user's
original unmodified scene_pickplace.xml.  The original 8 actuators, implicitfast
integrator, tendon/equality, meshes and inertias are therefore retained.

This is a custom robosuite environment, not robosuite's built-in Lift task.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from robosuite.environments.base import MujocoEnv
from robosuite.utils.binding_utils import MjSim


MODEL_DIR = Path(__file__).resolve().parents[1] / 'legacy_original_model' / 'franka_emika_panda'
SCENE_PATH = MODEL_DIR / 'scene_pickplace_stage3.xml'


class LegacyMenageriePandaRobosuite(MujocoEnv):
    """robosuite lifecycle, native Menagerie dynamics, legacy 8D actuators.

    This low-level robosuite env expects a callback to compute and write the
    next 7D DLS q-command and gripper target on *every* MuJoCo substep.
    It is wrapped by the exact old Gymnasium SAC task in panda_legacy_exact.py.
    """

    def __init__(self, scene_path: str | Path | None = None, control_period_s: float = 0.04):
        self.scene_path = Path(scene_path or SCENE_PATH).resolve()
        if not self.scene_path.is_file():
            raise FileNotFoundError(f'Original scene MJCF not found: {self.scene_path}')
        self.dls_callback = None
        freq = 1.0 / float(control_period_s)
        if freq <= 0 or abs(freq - round(freq)) > 1e-8:
            raise ValueError('control_period_s must correspond to an integer control frequency')
        # lite_physics=False calls MjSim.step()/mujoco.mj_step(), retaining the
        # MJCF implicitfast integrator; step1/step2 would change integrator.
        super().__init__(
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera=None,
            control_freq=int(round(freq)),
            lite_physics=False,
            horizon=100000,
            ignore_done=True,
            hard_reset=False,
            renderer='mujoco',
        )
        if int(self.sim.model.nu) != 8:
            raise RuntimeError(f'Original model must have 8 actuators, found {self.sim.model.nu}')
        if abs(self.model_timestep - float(self.sim.model.opt.timestep)) > 1e-12:
            raise RuntimeError('robosuite physics timestep differs from compiled Menagerie XML')
        n_steps = round(float(control_period_s) / self.model_timestep)
        if abs(n_steps * self.model_timestep - float(control_period_s)) > 1e-9:
            raise ValueError('control_period_s must be an integer multiple of model timestep')

    def _load_model(self):
        # Keep an identifiable, stable model handle. Actual compiled original
        # MJCF is exposed through self.sim.model._model, not a cloned robot XML.
        self.model = SimpleNamespace(mujoco_objects=[], source_path=str(self.scene_path))

    def _initialize_sim(self, xml_string=None):
        if xml_string is not None:
            raise NotImplementedError('This environment deliberately loads its original on-disk MJCF')
        compiled = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.sim = MjSim(compiled)
        self.sim.forward()
        self.initialize_time(self.control_freq)
        self.model_timestep = float(compiled.opt.timestep)
        self.control_timestep = 1.0 / self.control_freq

    def _setup_references(self):
        # The original XML uses stable original names; no robosuite RobotEnv
        # prefixing or object identity remapping should occur.
        return None

    def _setup_observables(self):
        from collections import OrderedDict
        return OrderedDict()

    def visualize(self, vis_settings):
        return None

    def _pre_action(self, action, policy_step=False):
        if self.dls_callback is None:
            raise RuntimeError('Original DLS substep callback has not been installed')
        self.dls_callback()

    def reward(self, action):
        # The outer original Gymnasium environment computes the ORIGINAL SAC
        # reward; the robosuite low-level engine cannot replace that reward.
        return 0.0

    def _check_success(self):
        return False

    @property
    def action_spec(self):
        m = self.sim.model._model
        return (np.asarray(m.actuator_ctrlrange[:, 0]).copy(),
                np.asarray(m.actuator_ctrlrange[:, 1]).copy())

    @property
    def action_dim(self):
        return int(self.sim.model.nu)

    def advance_one_policy_step(self):
        if self.dls_callback is None:
            raise RuntimeError('DLS callback missing')
        # This is the REAL robosuite MujocoEnv.step() / MjSim.step() path.
        # The action placeholder is ignored in _pre_action, where the original
        # DLS function writes all 8 original actuator setpoints.
        self.step(np.zeros(self.action_dim, dtype=np.float64))
