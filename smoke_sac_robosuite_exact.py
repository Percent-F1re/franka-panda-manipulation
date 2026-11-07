"""Light physical smoke test for original SAC / robosuite engine."""
from __future__ import annotations
import argparse
import numpy as np
import mujoco
from envs.panda_legacy_exact import PandaPickPlaceEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--steps',type=int,default=5)
    a=p.parse_args()
    env=PandaPickPlaceEnv(task='pickplace')
    try:
        obs,info=env.reset(seed=a.seed)
        assert obs.shape==(46,)
        assert env.action_space.shape==(4,)
        assert int(env.model.nu)==8
        assert int(env.raw_env.sim.model.nu)==8
        assert np.all(np.isfinite(obs))
        assert np.array_equal(env.raw_env.sim.data._data.qpos,env.data.qpos)
        assert env.raw_env.sim.data._data is env.data
        actual_int=int(env.model.opt.integrator)
        expected_int=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        assert actual_int==expected_int,(actual_int,expected_int)
        assert abs(env.dt-.002)<1e-10
        # Original home/gripper/geometry, not the new robosuite Panda model.
        assert np.allclose(env.data.qpos[env.qpos_ids],np.asarray([0.,0.,0.,-1.57079,0.,1.57079,-.7853]),atol=.02)
        assert np.isclose(env.model.actuator_ctrlrange[7,1],255.)
        print('[CHECK] True robosuite MujocoEnv subclass:',type(env.raw_env).__name__)
        print('[CHECK] Original Menagerie 8 actuators; implicitfast; shared MjData; 46D/4D')
        print('[CHECK] table surface z:',env.table_z,'cube xyz:',info['cube_xyz_m'])
        for i in range(a.steps):
            obs,rew,term,trunc,info=env.step(np.array([0.,0.,0.,1.],dtype=np.float32))
            assert obs.shape==(46,) and np.all(np.isfinite(obs))
            if term or trunc: break
        print('SMOKE PASSED: original Menagerie Panda running under robosuite MujocoEnv')
    finally:
        env.close()


if __name__=='__main__':main()
