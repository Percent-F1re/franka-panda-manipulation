"""Physics-safe domain randomization for the original Menagerie Panda task.

This extends the *selectable* DLS/OSC baseline without changing its reward,
observation, action, policy, DLS / OSC, event rules, or gripper.

All physics parameters are restored to their exact compiled baseline each reset,
then a new episode's mass/inertia/friction factors are applied BEFORE physical
settling. Goal XY is changed AFTER settling but BEFORE obs/reward initialization.
No mid-episode teleportations or reward hacks.

Physical design: 'friction_scale' simultaneously scales the sliding, torsional,
rolling coefficients of the cube, support table, and finger collision pads.
This controls actual contact friction under MuJoCo mixing; varying ONLY cube
friction can be masked by a higher partner geom's friction.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.panda_dual_controller import PandaDualControllerEnv


class RobustPandaDualControllerEnv(PandaDualControllerEnv):
    """4D/46D old-SAC compatible DLS/OSC with episode-wise randomization.

    options:
      xy_offset: [dx,dy] m, original cube XY randomization
      mass_scale: positive multiplier for cube body mass AND inertia
      friction_scale: positive multiplier for cube, table and finger pad friction
      target_delta_xy: [dx,dy] m, relative to original cube+TRANSFER_XY goal
    """

    def __init__(self, **kwargs):
        # Superclass calls reset during __init__. Do NOT alter nominal physics
        # until both controllers, MJCF and all indices are initialized.
        self._nominal_physics = None
        self._episode_randomization = None
        self._applied_randomization = None
        super().__init__(**kwargs)
        model = self.model
        self._affected_friction_geom_ids = self._identify_contact_geoms()
        self._nominal_physics = {
            'cube_mass': float(model.body_mass[self.cube]),
            'cube_inertia': model.body_inertia[self.cube, :].copy(),
            'friction': model.geom_friction[self._affected_friction_geom_ids, :].copy(),
        }
        # Base has already been reset with the ORIGINAL physics. No parameter
        # changes until an explicit next reset.

    def _identify_contact_geoms(self):
        model = self.model
        contact_geoms = {int(self.cube_geom), int(self.table_geom)}
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in (int(self.left), int(self.right)):
                continue
            if int(model.geom_contype[geom]) != 0 or int(model.geom_conaffinity[geom]) != 0:
                contact_geoms.add(geom)
        if len(contact_geoms) < 4:
            raise RuntimeError('Could not identify both collision finger pads; verify original Panda geoms')
        return np.array(sorted(contact_geoms), dtype=np.int32)

    @staticmethod
    def _two_vector(value, label):
        v = np.asarray(value, dtype=np.float64)
        if v.shape != (2,) or not np.all(np.isfinite(v)):
            raise ValueError(f'{label} must be a finite 2-element vector, got {value!r}')
        return v

    def reset(self, *, seed=None, options=None):
        data = dict(options or {})
        mass = float(data.pop('mass_scale', 1.0))
        friction = float(data.pop('friction_scale', 1.0))
        target = self._two_vector(data.pop('target_delta_xy', (0.0,0.0)), 'target_delta_xy')
        if not 0.25 <= mass <= 2.0 or not np.isfinite(mass):
            raise ValueError('mass_scale must be finite and between 0.25 and 2.0')
        if not 0.25 <= friction <= 2.0 or not np.isfinite(friction):
            raise ValueError('friction_scale must be finite and between 0.25 and 2.0')
        if np.any(np.abs(target) > 0.05 + 1e-12):
            raise ValueError('target_delta_xy must be within +/-0.05 m per component')
        self._episode_randomization = {
            'mass_scale': mass, 'friction_scale': friction,
            'target_delta_xy': target.copy(),
        }
        try:
            obs, info = super().reset(seed=seed, options=data)
            if self._nominal_physics is not None:
                info.update(self.randomization_info())
            return obs,info
        finally:
            self._episode_randomization = None

    def _before_original_settle(self):
        if self._nominal_physics is None:
            return
        p = self._episode_randomization or {
            'mass_scale':1.0, 'friction_scale':1.0,
            'target_delta_xy':np.zeros(2),
        }
        baseline=self._nominal_physics
        m=self.model
        # Restore exact compiled baseline first, avoiding cross-episode drift.
        m.body_mass[self.cube] = baseline['cube_mass']
        m.body_inertia[self.cube,:] = baseline['cube_inertia']
        m.geom_friction[self._affected_friction_geom_ids,:] = baseline['friction']
        m.body_mass[self.cube] *= p['mass_scale']
        m.body_inertia[self.cube,:] *= p['mass_scale']
        m.geom_friction[self._affected_friction_geom_ids,:] *= p['friction_scale']
        # MuJoCo cached inertia / kinematic constants must reflect new values.
        if not hasattr(mujoco,'mj_setConst'):
            raise RuntimeError('This MuJoCo version has no mj_setConst; cannot safely randomize inertia')
        mujoco.mj_setConst(m,self.data)
        self._applied_randomization = dict(
            mass_scale=float(p['mass_scale']),
            friction_scale=float(p['friction_scale']),
            target_delta_xy=np.asarray(p['target_delta_xy']).copy(),
        )

    def _after_original_target_set(self):
        if self._nominal_physics is None:
            return
        p = self._episode_randomization
        if p is None:
            return
        self.target_xy = self.target_xy + p['target_delta_xy']
        # Optional visual marker. This has no collision or task-physics effect.
        site_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'place_target_marker')
        if site_id >= 0:
            self.model.site_pos[site_id,:2] = self.target_xy
            mujoco.mj_forward(self.model,self.data)

    def randomization_info(self):
        p=self._applied_randomization
        if p is None:
            return {}
        return {
            'mass_scale':p['mass_scale'],
            'friction_scale':p['friction_scale'],
            'cube_mass_kg':float(self.model.body_mass[self.cube]),
            'target_delta_xy_m':np.asarray(p['target_delta_xy']).tolist(),
            'target_xy_m':self.target_xy.tolist(),
            'contact_friction_geoms':int(len(self._affected_friction_geom_ids)),
        }
