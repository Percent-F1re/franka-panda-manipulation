"""Windows robosuite 1.5.2 runtime validation for the selectable controllers.

Run this BEFORE any 100-episode comparison; checks model integrity, actuator
mode, 46D/4D dimensions, finite 6D OSC torques, and repeated-reset behavior.
"""
from __future__ import annotations
import argparse
import numpy as np
import mujoco
from envs.panda_dual_controller import PandaDualControllerEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--mode',choices=['both','dls','osc'],default='both')
    p.add_argument('--osc-kp',type=float,default=150.0)
    p.add_argument('--osc-damping-ratio',type=float,default=1.0)
    a=p.parse_args()
    if a.steps<1:p.error('steps must be positive')
    modes=['dls','osc'] if a.mode=='both' else [a.mode]
    initial={}
    for mode in modes:
        env=PandaDualControllerEnv(task='pickplace',controller=mode,
                                   osc_kp=a.osc_kp,
                                   osc_damping_ratio=a.osc_damping_ratio,
                                   max_episode_steps=600)
        try:
            for test_reset in (1,2):
                obs,info=env.reset(seed=a.seed,options={'xy_offset':[0.0,0.0]})
                assert obs.shape==(46,) and env.action_space.shape==(4,)
                assert np.all(np.isfinite(obs))
                assert env.model.nu==8
                assert env.data is env.raw_env.sim.data._data
                assert abs(float(env.model.opt.timestep)-.002)<1e-10
                if mode=='dls':
                    assert env._osc_bridge is None
                    assert int(env.model.actuator_gaintype[0])==int(mujoco.mjtGain.mjGAIN_FIXED)
                    assert float(env.model.actuator_gainprm[0,0])>1000.0
                    assert not env.controller_diagnostics()['torque_substeps']
                else:
                    assert env._osc_bridge.torque_enabled
                    assert env._osc_bridge.osc.name=='OSC_POSE'
                    assert int(env.model.actuator_biastype[0])==int(mujoco.mjtBias.mjBIAS_NONE)
                    assert abs(float(env.model.actuator_gainprm[0,0])-1.0)<1e-12
                    assert float(env.model.actuator_gainprm[7,0])!=1.0
                if test_reset==1:
                    initial[mode]=obs.copy()
                else:
                    assert np.max(np.abs(obs-initial[mode]))<1e-5, (mode,np.max(np.abs(obs-initial[mode])))
                for i in range(a.steps):
                    obs,reward,term,trunc,info=env.step(np.array([0.,0.,0.,1.],dtype=np.float32))
                    assert np.all(np.isfinite(obs)),(mode,i)
                    assert np.all(np.isfinite(env.data.ctrl)),(mode,i)
                    if term or trunc:break
                if mode=='osc':
                    assert env._osc_bridge.substeps>0
                    assert np.all(env.data.ctrl[:7] <= env._osc_bridge.tau_high+1e-9)
                    assert np.all(env.data.ctrl[:7] >= env._osc_bridge.tau_low-1e-9)
                print(f'[SMOKE] {mode.upper()} reset={test_reset} steps={i+1} '
                      f'scene={env.raw_env.scene_path} 4D/46D OK '
                      f'controller={env.controller_diagnostics()}',flush=True)
        finally:
            env.close()
    if len(modes)==2:
        assert np.allclose(initial['dls'],initial['osc'],rtol=0,atol=1e-5),\
            ('DLS vs OSC initial observation mismatch',float(np.max(np.abs(initial['dls']-initial['osc']))))
        print('[SMOKE] Initial observations match within 1e-5 across both modes')
    print('DUAL CONTROLLER SMOKE PASSED')

if __name__=='__main__':main()
