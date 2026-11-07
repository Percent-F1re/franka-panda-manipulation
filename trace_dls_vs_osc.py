"""Open-loop dynamics diagnostic: feed the SAME DLS-generated SAC actions to DLS and OSC.

This is NOT a closed-loop SAC evaluation and MUST NOT be reported as an OSC
success rate. It isolates actuator / controller dynamics at shared actions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from envs.panda_dual_controller import PandaDualControllerEnv
from legacy_sac_checkpoint import load_legacy_sac


def _physical_snapshot(env):
    hand,_=env._pose()
    cube=np.asarray(env.data.xpos[env.cube],dtype=float).copy()
    contact=env._contacts()
    return dict(hand=hand,cube=cube,
                aperture_mm=1000*sum(float(env.data.qpos[j]) for j in env.finger_qpos_ids),
                q=np.asarray(env.data.qpos[env.qpos_ids],dtype=float).copy(),
                grip_mm=1000*float(env._grasp_distance()),
                rise_mm=1000*float(cube[2]-env.cube_start[2]),
                contacts=contact)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--case',type=int,default=1)
    p.add_argument('--steps',type=int,default=220)
    p.add_argument('--osc-kp',type=float,default=150.0)
    p.add_argument('--osc-damping-ratio',type=float,default=1.0)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.case<1 or a.steps<1:p.error('--case and --steps must be >=1')
    out=Path(a.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):p.error('Use an empty --out')
    envs={}
    try:
        for mode in ('dls','osc'):
            envs[mode]=PandaDualControllerEnv(task='pickplace',controller=mode,
                                             max_episode_steps=600,
                                             osc_kp=a.osc_kp,osc_damping_ratio=a.osc_damping_ratio)
        model=load_legacy_sac(a.model,envs['dls'])
        offset=np.random.default_rng(a.seed).uniform(-.04,.04,size=(a.case,2))[-1]
        observations={}
        for mode in ('dls','osc'):
            observations[mode],_=envs[mode].reset(seed=a.seed+a.case-1,
                                                  options={'xy_offset':offset})
        out.mkdir(parents=True,exist_ok=True)
        initial={'case':a.case,'seed':a.seed,'xy_offset_m':offset.tolist(),
                 'initial_46d_max_abs_diff':float(np.max(np.abs(observations['dls']-observations['osc']))),
                 'controller':'original SAC actor from DLS observations for BOTH; diagnostic only'}
        (out/'initial.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
        print('[INITIAL]',initial,flush=True)
        rows=[]
        for t in range(a.steps):
            action,_=model.predict(observations['dls'],deterministic=True)
            info={};terminated={}
            for mode in ('dls','osc'):
                observations[mode],_,term,trunc,info[mode]=envs[mode].step(action)
                terminated[mode]=bool(term or trunc)
            d=_physical_snapshot(envs['dls']);o=_physical_snapshot(envs['osc'])
            r={'step':t+1,'action_x':float(action[0]),'action_y':float(action[1]),
               'action_z':float(action[2]),'action_grip':float(action[3]),
               'hand_xyz_delta_mm':1000*float(np.linalg.norm(d['hand']-o['hand'])),
               'cube_xyz_delta_mm':1000*float(np.linalg.norm(d['cube']-o['cube'])),
               'obs_46d_max_abs_delta':float(np.max(np.abs(observations['dls']-observations['osc']))),
               'q_delta_max_rad':float(np.max(np.abs(d['q']-o['q']))),
               'dls_grasp_dist_mm':d['grip_mm'],'osc_grasp_dist_mm':o['grip_mm'],
               'dls_gripper_aperture_mm':d['aperture_mm'],'osc_gripper_aperture_mm':o['aperture_mm'],
               'dls_cube_rise_mm':d['rise_mm'],'osc_cube_rise_mm':o['rise_mm'],
               'dls_left_contacts':d['contacts'][1],'osc_left_contacts':o['contacts'][1],
               'dls_right_contacts':d['contacts'][2],'osc_right_contacts':o['contacts'][2],
               'dls_success':bool(info['dls']['is_success']),
               'osc_success':bool(info['osc']['is_success']),
               'osc_torque_saturation_frac':envs['osc'].controller_diagnostics()['torque_saturation_fraction']}
            rows.append(r)
            if t==0 or (t+1) in (20,25,32,40,45,50) or (t+1)%25==0:
                print(f'[STEP {t+1}] hand_delta={r["hand_xyz_delta_mm"]:.3f}mm '
                      f'cube_rise=(dls {r["dls_cube_rise_mm"]:.2f}, osc {r["osc_cube_rise_mm"]:.2f})mm '
                      f'osc_sat={r["osc_torque_saturation_frac"]:.3f}',flush=True)
            if any(terminated.values()):
                print('[STOP] At least one environment terminated; no unsafe extra steps',flush=True)
                break
        with (out/'paired_step_trace.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary={'steps':len(rows),'max_hand_delta_mm':max(r['hand_xyz_delta_mm'] for r in rows),
                 'max_joint_delta_rad':max(r['q_delta_max_rad'] for r in rows),
                 'last':rows[-1]}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('SAVED',out,flush=True)
    finally:
        for env in envs.values():env.close()


if __name__=='__main__':main()
