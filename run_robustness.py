"""V3 zero-shot domain-randomization benchmark: original 46D/4D SAC, DLS vs OSC.

Seven predeclared scenarios, each with paired episodes and frozen SAC weights.
All randomized physics parameters are applied before the 1.5s settling step.
Results are appended to CSV after every episode; input scenarios + parameters
are saved before the first rollout. Never edits original checkpoint or XML.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import robosuite

from envs.panda_dual_controller import PandaDualControllerEnv
from envs.panda_domain_randomization import RobustPandaDualControllerEnv
from envs.dual_scene_core import SOURCE_ROBOT, SOURCE_SCENE
from legacy_dual_benchmark import eval_one
from legacy_sac_checkpoint import load_legacy_sac
from legacy_exact_rules import BenchmarkRules
from robustness_design import scenario_catalog,generate_cases,design_hash,summary


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()


def append_csv(path,row):
    isnew=not path.is_file() or path.stat().st_size==0
    with path.open('a',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row),extrasaction='raise')
        if isnew:w.writeheader()
        w.writerow(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='Original single 46D/4D SAC ZIP, read only')
    p.add_argument('--controller',choices=['dls','osc','both'],default='both')
    p.add_argument('--scenarios',default='all',help='all, or comma-separated names from robustness_scenarios.json')
    p.add_argument('--scenario-file',default=None,help='Optional custom JSON with same schema')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--case',type=int,default=None,help='Only the 1-based case, same across scenarios')
    p.add_argument('--max-episode-steps',type=int,default=600)
    p.add_argument('--osc-kp',type=float,default=150.0)
    p.add_argument('--osc-damping-ratio',type=float,default=1.0)
    p.add_argument('--post-hold',choices=['target','raw_actuator'],default='target')
    p.add_argument('--save-traces',action='store_true')
    p.add_argument('--out',required=True,help='NEW empty output folder, never overwrite')
    a=p.parse_args()
    if a.episodes<1 or a.max_episode_steps<1 or a.seed<0 or a.seed>2**32-1:
        p.error('episodes and max-episode-steps must be positive; seed 0..2^32-1')
    if a.case is not None and a.case<1:p.error('--case must be >=1')
    if a.osc_kp<=0 or a.osc_damping_ratio<=0:p.error('OSC controller parameters must be positive')
    out=Path(a.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):p.error(f'Output folder is not empty: {out}')
    model_path=Path(a.model).expanduser().resolve()
    if not model_path.is_file():p.error(f'Checkpoint missing: {model_path}')
    catalog=scenario_catalog(a.scenario_file) if a.scenario_file else scenario_catalog()
    scenarios=list(catalog) if a.scenarios=='all' else [s.strip() for s in a.scenarios.split(',')]
    if not scenarios or len(set(scenarios))!=len(scenarios) or any(s not in catalog for s in scenarios):
        p.error(f'Unknown/duplicate scenarios; available: {list(catalog)}')
    modes=['dls','osc'] if a.controller=='both' else [a.controller]
    n_cases=max(a.episodes,a.case or 0)
    ids=[a.case] if a.case is not None else list(range(1,a.episodes+1))
    designs={s:generate_cases(a.seed,n_cases,s,catalog[s]) for s in scenarios}
    model=None
    rows=[]
    try:
        for sc in scenarios:
            conf=catalog[sc]
            envs={}
            try:
                for mode in modes:
                    klass=PandaDualControllerEnv if sc=='baseline' else RobustPandaDualControllerEnv
                    envs[mode]=klass(
                        controller=mode,task='pickplace',
                        random_xy_range_m=max(.04,float(conf['object_xy_half_range_m'])),
                        max_episode_steps=a.max_episode_steps,
                        osc_kp=a.osc_kp,osc_damping_ratio=a.osc_damping_ratio,
                    )
                    assert envs[mode].action_space.shape==(4,) and envs[mode].observation_space.shape==(46,)
                    assert envs[mode].model.nu==8
                if model is None:
                    model=load_legacy_sac(model_path,envs[modes[0]])
                    # Initialize output only after all controller/Actor checks pass.
                    out.mkdir(parents=True,exist_ok=True)
                    metadata={
                        **vars(a),'model':str(model_path),
                        'model_sha256':hash_file(model_path),
                        'robot_xml_sha256':hash_file(SOURCE_ROBOT),
                        'scene_xml_sha256':hash_file(SOURCE_SCENE),
                        'robosuite_version':str(getattr(robosuite,'__version__','unknown')),
                        'mujoco_version':str(getattr(mujoco,'__version__','unknown')),
                        'physical_rules':vars(BenchmarkRules()),
                        'osc_semantics':'official robosuite OSC torque controller on seven original Panda joint transmissions',
                        'dls_semantics':'original Menagerie joint position servos + original 6D DLS',
                        'actor':'one unchanged original DLS-trained SAC, independent closed-loop per controller',
                        'friction_semantics':'common multiplier on original cube/table/finger collision geoms, all three coefficients',
                        'mass_semantics':'cube body mass and all principal moments of inertia scaled together',
                        'target_semantics':'cube initial position + [0,+.06]m + target_delta_xy',
                        'post_hold_semantics':'target hold (primary) or raw actuator hold (secondary)',
                        'baseline_compatibility':'baseline uses exact original dual-controller environment and legacy XY offset PRNG',
                        'scenarios':{s:catalog[s] for s in scenarios},
                        'design_sha256':{s:design_hash(designs[s]) for s in scenarios},
                    }
                    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
                    (out/'design.json').write_text(json.dumps(
                        {s:[designs[s][i-1] for i in ids] for s in scenarios},
                        indent=2,ensure_ascii=False),encoding='utf-8')
                print(f'=== SCENARIO {sc}: {conf["description"]} ===',flush=True)
                for i in ids:
                    case=designs[sc][i-1]
                    for mode in modes:
                        episode_options={} if sc=='baseline' else {
                            'mass_scale':case['mass_scale'],
                            'friction_scale':case['friction_scale'],
                            'target_delta_xy':case['target_delta_xy'],
                        }
                        r,trace=eval_one(
                            model,envs[mode],seed=case['seed'],
                            offset=case['xy_offset'],post_hold=a.post_hold,
                            save_trace=a.save_traces, episode_options=episode_options,
                        )
                        # An explicit line per episode, including all physical factors.
                        result={
                            'scenario':sc,'index':int(i),'seed':int(case['seed']),
                            'initial_cube_offset_x_m':case['xy_offset'][0],
                            'initial_cube_offset_y_m':case['xy_offset'][1],
                            'mass_scale':case['mass_scale'],
                            'cube_mass_kg':float(envs[mode].model.body_mass[envs[mode].cube]),
                            'friction_scale':case['friction_scale'],
                            'target_delta_x_m':case['target_delta_xy'][0],
                            'target_delta_y_m':case['target_delta_xy'][1],
                            'target_x_m':float(envs[mode].target_xy[0]),
                            'target_y_m':float(envs[mode].target_xy[1]),
                            **r,
                        }
                        rows.append(result)
                        append_csv(out/'episodes.csv',result)
                        if a.save_traces and trace:
                            trace_dir=out/'traces'/sc
                            trace_dir.mkdir(parents=True,exist_ok=True)
                            trace_path=trace_dir/f'{mode}_case_{i:03d}.csv'
                            with trace_path.open('w',encoding='utf-8-sig',newline='') as f:
                                w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
                        print(f'[{sc:>22}] case {i}/{len(ids)} {mode.upper():>3} '
                              f'unified={r["unified_success"]} lift={r["unified_lift"]} '
                              f'transfer={r["unified_transfer"]} '
                              f'final_xy={r["final_xy_error_mm"]:.2f}mm',flush=True)
                    # Summaries written after each *paired* case, safe on Ctrl+C.
                    (out/'summary.json').write_text(json.dumps(
                        summary(rows,scenarios,modes),indent=2,ensure_ascii=False),encoding='utf-8')
            finally:
                for e in envs.values():e.close()
        print('FINAL SUMMARY:',json.dumps(summary(rows,scenarios,modes),ensure_ascii=False,indent=2),flush=True)
        print('FILES:',out,flush=True)
    except KeyboardInterrupt:
        print('Interrupted: completed episode rows and current summary remain on disk',flush=True)
        raise


if __name__=='__main__':main()
