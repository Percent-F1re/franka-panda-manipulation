"""Paired closed-loop same-SAC evaluation: Menagerie DLS vs robosuite OSC_POSE.

For each index, choose ONE identical XY offset and seed, then run the same
unchanged pretrained 46D->4D SAC actor independently in each controller env.
Common reward-independent physical success rules; same 1s post-target hold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import mujoco
import robosuite

from envs.panda_dual_controller import PandaDualControllerEnv
from legacy_sac_checkpoint import load_legacy_sac
from legacy_dual_benchmark import eval_one
from legacy_exact_rules import wilson_interval, BenchmarkRules
from envs.dual_scene_core import SOURCE_ROBOT, SOURCE_SCENE


def _append(path:Path,record:dict):
    new=not path.is_file() or path.stat().st_size==0
    with path.open('a',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(record.keys()),extrasaction='raise')
        if new:w.writeheader()
        w.writerow(record)


def _sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _paired_mcnemar_pvalue(b,c):
    """Exact two-sided binomial McNemar p-value, for n<=1000 pairs."""
    n=b+c
    if not n:return 1.0
    k=min(b,c)
    return min(1.0,2.0*sum(math.comb(n,i) for i in range(k+1))/2**n)


def _summarize(rows,mode):
    x=[r for r in rows if r['controller']==mode]
    n=len(x)
    s=sum(bool(r['unified_success']) for r in x)
    return {
        'n':n,'unified_successes':s,'unified_success_rate':s/n if n else None,
        'wilson_95ci':wilson_interval(s,n) if n else [None,None],
        'original_successes':sum(bool(r['original_is_success']) for r in x),
        'unified_lift':sum(bool(r['unified_lift']) for r in x),
        'unified_transfer':sum(bool(r['unified_transfer']) for r in x),
        'mean_final_xy_error_mm':float(np.mean([r['final_xy_error_mm'] for r in x])) if n else None,
        'mean_max_hand_target_error_mm':float(np.mean([r['max_hand_target_error_mm'] for r in x])) if n else None,
        'total_torque_saturated_substeps':sum(int(r['torque_saturated_substeps'] or 0) for r in x),
        'total_osc_torque_substeps':sum(int(r['torque_substeps'] or 0) for r in x),
        'fail_reasons':{
            k:sum(not bool(r[k]) for r in x)
            for k in ('xy_ok','z_ok','table_contact_ok',
                      'fingers_released_ok','speed_ok','drift_ok')
        },
    }


def _make_summary(rows,requested):
    report={mode:_summarize(rows,mode) for mode in requested}
    if set(requested)=={'dls','osc'}:
        d={int(r['index']):r for r in rows if r['controller']=='dls'}
        o={int(r['index']):r for r in rows if r['controller']=='osc'}
        ids=sorted(set(d)&set(o))
        both=sum(d[i]['unified_success'] and o[i]['unified_success'] for i in ids)
        d_only=sum(d[i]['unified_success'] and not o[i]['unified_success'] for i in ids)
        o_only=sum(o[i]['unified_success'] and not d[i]['unified_success'] for i in ids)
        neither=sum(not d[i]['unified_success'] and not o[i]['unified_success'] for i in ids)
        report['paired']={
            'n':len(ids),
            'both_success':both,
            'dls_only_success':d_only,
            'osc_only_success':o_only,
            'neither_success':neither,
            'absolute_success_rate_difference_dls_minus_osc':(
                (d_only-o_only)/len(ids) if ids else None),
            'exact_mcnemar_p_two_sided':_paired_mcnemar_pvalue(d_only,o_only),
        }
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='ORIGINAL 46D/4D SAC ZIP; read only')
    p.add_argument('--controller',choices=['dls','osc','both'],default='both')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--case',type=int,default=None,help='Optional 1-based case index from this seeded offset list')
    p.add_argument('--max-episode-steps',type=int,default=600)
    p.add_argument('--osc-kp',type=float,default=150.0)
    p.add_argument('--osc-damping-ratio',type=float,default=1.0)
    p.add_argument('--post-hold',choices=['target','raw_actuator'],default='target')
    p.add_argument('--save-traces',action='store_true')
    p.add_argument('--out',required=True,help='A NEW empty output directory; never overwrite data')
    args=p.parse_args()
    if args.episodes<1 or args.max_episode_steps<1 or (args.case is not None and args.case<1):
        p.error('--episodes, --max-episode-steps and --case must be positive')
    if args.osc_kp<=0 or args.osc_damping_ratio<=0:
        p.error('--osc-kp and --osc-damping-ratio must be positive')
    out=Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):p.error(f'Output directory must be new/empty: {out}')
    model_path=Path(args.model).expanduser().resolve()
    modes=['dls','osc'] if args.controller=='both' else [args.controller]
    n_offsets=max(args.episodes,args.case or 0)
    offsets=np.random.default_rng(args.seed).uniform(-.04,.04,size=(n_offsets,2))
    cases=[args.case] if args.case is not None else list(range(1,args.episodes+1))
    envs={}
    rows=[]
    try:
        for mode in modes:
            envs[mode]=PandaDualControllerEnv(
                controller=mode, task='pickplace',
                max_episode_steps=args.max_episode_steps,
                osc_kp=args.osc_kp,osc_damping_ratio=args.osc_damping_ratio,
            )
            assert tuple(envs[mode].action_space.shape)==(4,)
            assert tuple(envs[mode].observation_space.shape)==(46,)
            assert envs[mode].model.nu==8
        model=load_legacy_sac(model_path,envs[modes[0]])
        out.mkdir(parents=True,exist_ok=True)
        metadata={
            **vars(args),
            'original_actor_sha256':_sha256(model_path),
            'original_robot_xml_sha256':_sha256(SOURCE_ROBOT),
            'original_stage3_scene_sha256':_sha256(SOURCE_SCENE),
            'model':'original Menagerie Panda, both arms; 8 original actuators',
            'robosuite_version':str(getattr(robosuite,'__version__','unknown')),
            'mujoco_version':str(getattr(mujoco,'__version__','unknown')),
            'policy_observation_shape':[46],
            'policy_action_shape':[4],
            'action_semantics':'+/-1 cm XYZ increment per 0.04s; gripper +1=open/-1=close',
            'osc_implementation':'robosuite.controllers.parts.arm.osc.OperationalSpaceController (OSC_POSE)',
            'osc_actuator_difference':'Only arm general actuators switch from position servos to unit-gain torque motors; original forcerange preserved',
            'osc_q_cmd_46d_feature':'shadow DLS command as observation compatibility feature; not executed',
            'post_hold_semantics':{
                'target':'freeze final Cartesian target and gripper; DLS holds position servos, OSC continues torque feedback',
                'raw_actuator':'freeze actuator ctrl values for both; not fair to torque OSC; secondary sensitivity only',
            }[args.post_hold],
            'physical_rules':vars(BenchmarkRules()),
            'xy_offset_distribution':'uniform [-0.04, +0.04] per axis; same offsets in both modes',
        }
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        for idx in cases:
            offset=offsets[idx-1]
            for mode in modes:
                # Each mode receives the same seed/offset; each receives its
                # OWN observations and chooses its OWN deterministic SAC actions.
                r,trace=eval_one(model,envs[mode],seed=args.seed+idx-1,
                                 offset=offset,post_hold=args.post_hold,
                                 save_trace=args.save_traces)
                record={'index':idx,'offset_x_m':float(offset[0]),
                        'offset_y_m':float(offset[1]),**r}
                rows.append(record)
                _append(out/f'episodes_{mode}.csv',record)
                if trace:
                    name=f'{mode}_case_{idx:03d}_trace.csv'
                    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(trace[0]))
                        w.writeheader();w.writerows(trace)
                print(f'[{idx:03d}/{len(cases):03d}] {mode.upper()} '
                      f'unified={r["unified_success"]} lift={r["unified_lift"]} '
                      f'transfer={r["unified_transfer"]} '
                      f'XY={r["final_xy_error_mm"]:.1f}mm '
                      f'sat={r["torque_saturation_fraction"]}',flush=True)
            if set(modes) == {'dls', 'osc'}:
                same_idx = {r['controller']: r for r in rows if r['index'] == idx}
                dr, orow = same_idx['dls'], same_idx['osc']
                pair = {
                    'index': idx,
                    'offset_x_m': float(offset[0]),
                    'offset_y_m': float(offset[1]),
                    'dls_success': bool(dr['unified_success']),
                    'osc_success': bool(orow['unified_success']),
                    'dls_lift': bool(dr['unified_lift']),
                    'osc_lift': bool(orow['unified_lift']),
                    'dls_transfer': bool(dr['unified_transfer']),
                    'osc_transfer': bool(orow['unified_transfer']),
                    'dls_xy_error_mm': float(dr['final_xy_error_mm']),
                    'osc_xy_error_mm': float(orow['final_xy_error_mm']),
                    'osc_torque_saturation_fraction': orow['torque_saturation_fraction'],
                }
                _append(out/'paired.csv', pair)
            report=_make_summary(rows,modes)
            (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print('\nFINAL SUMMARY',json.dumps(_make_summary(rows,modes),indent=2,ensure_ascii=False))
        print('SAVED:',out)
    finally:
        for env in envs.values():
            env.close()


if __name__=='__main__':main()
