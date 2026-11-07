"""Original single SAC, unchanged Menagerie scene, *robosuite MujocoEnv* runner.

By default use original benchmark's paired uniform XY-offset generator.
Always evaluate with deterministic Actor, original reward-independent success,
and 1 second of frozen post-episode physics. Saves each episode immediately.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from envs.panda_legacy_exact import PandaPickPlaceEnv
from legacy_sac_checkpoint import load_legacy_sac
from legacy_exact_benchmark import evaluate_episode
from legacy_exact_rules import BenchmarkRules, wilson_interval


def _append_csv(path: Path, d: dict):
    with path.open('a', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(d))
        if f.tell() == 0: w.writeheader()
        w.writerow(d)


def main():
    p=argparse.ArgumentParser(description='Unchanged original SAC checkpoint / robosuite Menagerie scene')
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=42000, help='Original benchmark XY offset generator seed')
    p.add_argument('--max-episode-steps', type=int, default=600)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--save-traces', action='store_true')
    p.add_argument('--case', type=int, default=None, help='Only check the 1-based case from the seeded offset list')
    args=p.parse_args()
    if args.episodes < 1 or args.max_episode_steps < 1 or (args.case is not None and args.case<1):
        p.error('episodes, case and max-episode-steps must be positive')
    out = Path(args.out).expanduser().resolve() if args.out else Path(__file__).resolve().parent / 'evaluation' / 'exact_menagerie' / f'seed{args.seed}_n{args.episodes}'
    if not args.check_only and out.exists() and any(out.iterdir()):
        p.error(f'Output directory not empty: {out}. Use a NEW --out')
    env=PandaPickPlaceEnv(task='pickplace',max_episode_steps=args.max_episode_steps)
    try:
        model=load_legacy_sac(args.model, env, args.device)
        if args.check_only:
            print(f'[CHECK PASSED] robosuite MujocoEnv={type(env.raw_env).__name__} '
                  f'original MJCF actuators={env.model.nu}; 46D/4D SAC loads',flush=True)
            print('Integrator:',int(env.model.opt.integrator),
                  'physics dt:',float(env.model.opt.timestep), 'control dt:',env.control_dt)
            return
        out.mkdir(parents=True,exist_ok=True)
        n=max(args.episodes,args.case or 0)
        offsets=np.random.default_rng(args.seed).uniform(-env.random_xy_range_m,env.random_xy_range_m,size=(n,2))
        cases=[args.case] if args.case is not None else list(range(1,args.episodes+1))
        metadata={'environment':'robosuite.environments.base.MujocoEnv custom class',
                  'model':'original Menagerie panda_pickplace.xml + scene_pickplace.xml',
                  'original_sac_checkpoint':str(Path(args.model).resolve()),
                  'seed':args.seed,'episodes':len(cases), 'offset_distribution':'+/-0.04m XY uniform',
                  'model_nu':int(env.model.nu),'dt':float(env.dt),'control_dt':env.control_dt,
                  'post_observation_s':1.0}
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        rows=[]
        for idx in cases:
            case_seed=args.seed+idx-1
            record, trace=evaluate_episode(model,env,seed=case_seed,xy_offset=offsets[idx-1],
                                           save_trace=args.save_traces)
            record={'index':idx,'offset_x_m':float(offsets[idx-1][0]),
                    'offset_y_m':float(offsets[idx-1][1]),**record}
            rows.append(record)
            _append_csv(out/'episodes.csv',record)
            if trace:
                with (out/f'case_{idx:03d}_trace.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
            successes=sum(r['unified_success'] for r in rows)
            print(f'[{len(rows):03d}/{len(cases):03d}] idx={idx} seed={case_seed} '
                  f'original={record["original_is_success"]} unified={record["unified_success"]} '
                  f'xy={record["final_xy_error_mm"]:.2f}mm total={successes}/{len(rows)}',flush=True)
        successes=sum(r['unified_success'] for r in rows)
        summary={'n':len(rows),'successes':successes,'success_rate':successes/len(rows),
                 'original_successes':sum(r['original_is_success'] for r in rows),
                 'unified_lift':sum(r['unified_lift'] for r in rows),
                 'unified_transfer':sum(r['unified_transfer'] for r in rows),
                 'wilson_95ci':wilson_interval(successes,len(rows))}
        (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        print('RESULT',json.dumps(summary,ensure_ascii=False),'\nSAVED',out,flush=True)
    finally:
        env.close()


if __name__=='__main__': main()
