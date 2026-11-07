"""Old native-MuJoCo vs robosuite-MujocoEnv, SAME original actor actions.

Uses old observation for both environments to isolate dynamics from policy feedback.
Use --old-env to point to the *original* PandaPickPlaceEnv.py on your Windows PC.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from envs.panda_legacy_exact import PandaPickPlaceEnv as RobosuiteExactEnv
from legacy_sac_checkpoint import load_legacy_sac


def load_native_env(path):
    path=Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('original_sac_environment_for_comparison',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PandaPickPlaceEnv


def state(env):
    cube=np.asarray(env.data.xpos[env.cube],dtype=float)
    hand=np.asarray(env.data.xpos[env.hand],dtype=float)
    tbl, left,right=env._contacts()
    return dict(
        hand_xyz=hand,cube_xyz=cube,
        q=np.asarray(env.data.qpos[env.qpos_ids],dtype=float).copy(),
        aperture_mm=1000*sum(float(env.data.qpos[k]) for k in env.finger_qpos_ids),
        rise_mm=1000*float(cube[2]-env.cube_start[2]),
        grasp_mm=1000*float(env._grasp_distance()),
        contacts=(tbl,left,right),
        q_tracking=float(np.max(np.abs(env.q_cmd-env.data.qpos[env.qpos_ids]))),
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--old-env',required=True,help='Original robot_simulation/sac_pickplace/panda_pickplace_env.py')
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--index',type=int,default=1)
    p.add_argument('--steps',type=int,default=220)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    if a.index<1 or a.steps<1: p.error('--index, --steps must be >=1')
    out=Path(a.out).expanduser().resolve() if a.out else Path(__file__).resolve().parent/'evaluation'/'exact_menagerie'/f'compare_seed{a.seed}_case{a.index:03d}'
    if out.exists() and any(out.iterdir()):p.error(f'Output directory exists: {out}. Use NEW --out')
    out.mkdir(parents=True,exist_ok=True)
    # Stage4 of the user's original project changes cwd at import; paths above
    # have been resolved and the robosuite model is loaded from an absolute path.
    Original=load_native_env(a.old_env)
    old=Original(task='pickplace',max_episode_steps=600)
    new=RobosuiteExactEnv(task='pickplace',max_episode_steps=600)
    try:
        model=load_legacy_sac(a.model,new)
        offsets=np.random.default_rng(a.seed).uniform(-.04,.04,size=(a.index,2))
        off=offsets[-1]
        old_obs,old_info=old.reset(seed=a.seed+a.index-1,options={'xy_offset':off})
        new_obs,new_info=new.reset(seed=a.seed+a.index-1,options={'xy_offset':off})
        obs_delta=np.abs(np.asarray(old_obs)-np.asarray(new_obs))
        init={'max_abs_46d':float(np.max(obs_delta)),
              'max_abs_joint_q':float(np.max(obs_delta[:7])),
              'max_abs_qcmd':float(np.max(obs_delta[14:21])),
              'max_abs_hand_xyz_m':float(np.max(obs_delta[21:24])),
              'max_abs_cube_xyz_m':float(np.max(obs_delta[27:30])),
              'offset_xy_m':off.tolist(),
              'original_xml_actuators':int(old.model.nu),
              'robosuite_xml_actuators':int(new.model.nu),
              'old_dt':float(old.dt),'robosuite_dt':float(new.dt)}
        (out/'initial_comparison.json').write_text(json.dumps(init,indent=2),encoding='utf-8')
        print('[INITIAL]',json.dumps(init),flush=True)
        rows=[]
        old_done=new_done=False
        for i in range(a.steps):
            # Crucially use SAME ACTOR ACTION from OLD observation on both.
            action,_=model.predict(old_obs,deterministic=True)
            old_obs,_,ot,oc,oi=old.step(action)
            new_obs,_,nt,nc,ni=new.step(action)
            os0=state(old); ns0=state(new)
            row={'step':i+1,'action_x':float(action[0]),'action_y':float(action[1]),
                 'action_z':float(action[2]),'action_grip':float(action[3]),
                 'hand_xyz_delta_mm':float(1000*np.linalg.norm(os0['hand_xyz']-ns0['hand_xyz'])),
                 'cube_xyz_delta_mm':float(1000*np.linalg.norm(os0['cube_xyz']-ns0['cube_xyz'])),
                 'q_delta_max_rad':float(np.max(np.abs(os0['q']-ns0['q']))),
                 'obs_46d_delta_max':float(np.max(np.abs(old_obs-new_obs))),
                 'old_finger_aperture_mm':float(os0['aperture_mm']),
                 'new_finger_aperture_mm':float(ns0['aperture_mm']),
                 'old_grasp_dist_mm':float(os0['grasp_mm']),
                 'new_grasp_dist_mm':float(ns0['grasp_mm']),
                 'old_rise_mm':float(os0['rise_mm']),
                 'new_rise_mm':float(ns0['rise_mm']),
                 'old_table_contacts':int(os0['contacts'][0]),
                 'new_table_contacts':int(ns0['contacts'][0]),
                 'old_left_contacts':int(os0['contacts'][1]),
                 'new_left_contacts':int(ns0['contacts'][1]),
                 'old_right_contacts':int(os0['contacts'][2]),
                 'new_right_contacts':int(ns0['contacts'][2]),
                 'old_tracking_max_rad':float(os0['q_tracking']),
                 'new_tracking_max_rad':float(ns0['q_tracking']),
                 'old_lift':bool(oi['has_lifted']),
                 'new_lift':bool(ni['has_lifted']),
                 'old_transfer':bool(oi['has_transferred']),
                 'new_transfer':bool(ni['has_transferred']),
                 'old_success':bool(oi['is_success']),
                 'new_success':bool(ni['is_success'])}
            rows.append(row)
            if i in (0,19,24,31,39,44,49) or (i+1)%25==0:
                print(f'[STEP {i+1}] hand_delta={row["hand_xyz_delta_mm"]:.3f}mm '
                      f'finger=({row["old_finger_aperture_mm"]:.2f}, {row["new_finger_aperture_mm"]:.2f})mm '
                      f'rise=({row["old_rise_mm"]:.2f}, {row["new_rise_mm"]:.2f})mm',flush=True)
            if ot or oc or nt or nc:
                print('[STOP] One env finished; no stepping after terminal/truncation',flush=True)
                break
        with (out/'paired_step_trace.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary={'completed_steps':len(rows),
                 'max_hand_delta_mm':max(r['hand_xyz_delta_mm'] for r in rows),
                 'max_joint_delta_rad':max(r['q_delta_max_rad'] for r in rows),
                 'old_success':rows[-1]['old_success'],
                 'new_success':rows[-1]['new_success'],
                 'old_lift':rows[-1]['old_lift'],'new_lift':rows[-1]['new_lift'],
                 'end_aperture_old_mm':rows[-1]['old_finger_aperture_mm'],
                 'end_aperture_new_mm':rows[-1]['new_finger_aperture_mm']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('SUMMARY',json.dumps(summary),'\nSAVED',out,flush=True)
    finally:
        old.close();new.close()


if __name__=='__main__':main()
