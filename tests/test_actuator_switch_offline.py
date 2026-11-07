"""Offline functional test: 7 motor actuators switch; gripper untouched."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ActuatorSwitchOffline(unittest.TestCase):
    def test_motor_conversion_and_exact_restoration(self):
        fake_m=types.ModuleType('mujoco')
        fake_m.mjtObj=types.SimpleNamespace(mjOBJ_ACTUATOR=1,mjOBJ_SITE=2)
        fake_m.mjtTrn=types.SimpleNamespace(mjTRN_JOINT=5)
        fake_m.mjtGain=types.SimpleNamespace(mjGAIN_FIXED=0)
        fake_m.mjtBias=types.SimpleNamespace(mjBIAS_NONE=0)
        fake_m.mj_id2name=lambda m,typ,i:f'actuator{i+1}'
        fake_m.mj_name2id=lambda m,typ,name:1 if typ==2 else int(name.removeprefix('actuator'))-1
        fake_m.mj_forward=lambda m,data:None
        fake_scene=types.ModuleType('envs.dual_scene_core')
        fake_scene.EEF_SITE='dual_osc_hand_ref'
        fake_scene.CENTER_SITE='hand_center'
        old_m=sys.modules.get('mujoco')
        old_s=sys.modules.get('envs.dual_scene_core')
        sys.modules['mujoco']=fake_m
        sys.modules['envs.dual_scene_core']=fake_scene
        try:
            spec=importlib.util.spec_from_file_location('offline_osc',ROOT/'envs/controllers/osc_bridge.py')
            mod=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            m=types.SimpleNamespace(
                nu=8,
                actuator_gaintype=np.ones(8,dtype=int),
                actuator_biastype=np.ones(8,dtype=int),
                actuator_gainprm=np.full((8,10),4500.,dtype=float),
                actuator_biasprm=np.full((8,10),-100.,dtype=float),
                actuator_ctrlrange=np.array([[-2,2.]]*7+[[0,255.]]),
                actuator_ctrllimited=np.ones(8,dtype=int),
                actuator_forcerange=np.array([[-87,87.]]*4+[[-12,12.]]*3+[[-100,100.]]),
                actuator_forcelimited=np.ones(8,dtype=int),
                actuator_gear=np.tile([1,0,0,0,0,0.],(8,1)),
                actuator_trntype=np.full(8,5,dtype=int),
            )
            orig={name:getattr(m,name).copy() for name in (
                'actuator_gaintype','actuator_biastype','actuator_gainprm',
                'actuator_biasprm','actuator_ctrlrange','actuator_ctrllimited')}
            e=types.SimpleNamespace(model=m,data=types.SimpleNamespace(ctrl=np.zeros(8),qpos=np.zeros(7)),
                                    qpos_ids=np.arange(7),qlo=np.full(7,-2.),qhi=np.full(7,2.))
            bridge=mod.RobosuiteOSCBridge(e,lambda:None)
            bridge.enter_torque_mode()
            np.testing.assert_array_equal(m.actuator_gainprm[:7,0],np.ones(7))
            np.testing.assert_array_equal(m.actuator_ctrlrange[:7],m.actuator_forcerange[:7])
            for name, old in orig.items():
                np.testing.assert_array_equal(getattr(m,name)[7],old[7])
            bridge.exit_torque_mode()
            for name, old in orig.items():
                np.testing.assert_array_equal(getattr(m,name),old)
        finally:
            if old_m is None:sys.modules.pop('mujoco',None)
            else:sys.modules['mujoco']=old_m
            if old_s is None:sys.modules.pop('envs.dual_scene_core',None)
            else:sys.modules['envs.dual_scene_core']=old_s

if __name__=='__main__':unittest.main()
