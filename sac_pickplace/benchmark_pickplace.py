"""Paired, reward-independent Franka Panda Pick-and-Place benchmark.

Put this file and benchmark_common.py in robot_simulation/sac_pickplace,
next to the already-working panda_pickplace_env.py. Keep Stage1-4 in parent.

Run (from sac_pickplace):
  python benchmark_pickplace.py --model results_pickplace/seed_42/best_model.zip --episodes 10 --seed 30000
  python benchmark_pickplace.py --model results_pickplace/seed_42/best_model.zip --episodes 100 --seed 40000 --out results_benchmark/final_100
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC

from benchmark_common import BenchmarkRules, audit_events, audit_final, sample_state, summarize
from panda_pickplace_env import PandaPickPlaceEnv

HERE = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ids_for(s4):
    return {
        'cube': s4.cube, 'cube_geom': s4.cube_geom, 'table_geom': s4.table_geom,
        'left': s4.left, 'right': s4.right, 'cube_dof_addr': s4.cube_dof_addr,
        'cube_half_width': float(s4.CUBE_HALF_WIDTH),
        'contact_fn': s4.count_cube_contacts,
    }


def expert_logs_to_samples(path: Path, cube_start: np.ndarray, rules: BenchmarkRules):
    """Convert Stage 4's ~100Hz physical telemetry into the shared auditor format."""
    result = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            xyz = np.array([float(row[f'cube_{axis}_m']) for axis in 'xyz'], dtype=float)
            target_xy = np.array([float(row['target_x_m']), float(row['target_y_m'])])
            # Stage4 does not log the cube bottom gap, but the shared final audit
            # reads that from physics directly after the identical 1s post window.
            result.append({
                'time_s': float(row['time_s']),
                'cube_xyz': xyz,
                'rise_mm': float((xyz[2] - cube_start[2]) * 1000),
                'horizontal_moved_mm': float(np.linalg.norm(xyz[:2]-cube_start[:2]) * 1000),
                'xy_error_mm': float(np.linalg.norm(xyz[:2]-target_xy) * 1000),
                'table_contacts': int(row['table_contacts']),
                'left_contacts': int(row['left_contacts']),
                'right_contacts': int(row['right_contacts']),
            })
    return result


def post_observe(model, data, ids, cube_start, target_xy, duration_s: float):
    """Exactly the same intervention for expert and SAC: hold current commands.

    We do NOT force the fingers open or change the arm target here: that could
    artificially rescue a failed policy. The robot's final controls are frozen.
    """
    physics_dt = float(model.opt.timestep)
    n = max(1, int(round(duration_s / physics_dt)))
    controls = data.ctrl.copy()
    samples = []
    for j in range(n):
        data.ctrl[:] = controls
        mujoco.mj_step(model, data)
        samples.append(sample_state(
            model, data, ids=ids, cube_initial_xyz=cube_start,
            target_xy=target_xy, time_s=float(data.time)
        ))
    return samples


def evaluate_expert(s4, ids, offset, episode_index, rules, logs_dir, keep_logs):
    result = s4.run_episode(episode_index, offset, save_expert=False, save_log=True)
    log_path = logs_dir / 'episode_logs' / f'episode_{episode_index:03d}.csv'
    if not log_path.is_file():
        raise RuntimeError(f'Expert did not save physical trace: {log_path}')
    # The first Stage4 telemetry sample is taken immediately after the first
    # physics step and reflects the physically settled cube pose. This is a
    # more faithful reference height than the nominal table + half-width.
    with log_path.open('r', encoding='utf-8-sig', newline='') as log_file:
        first = next(csv.DictReader(log_file), None)
    if first is None:
        raise RuntimeError(f'Empty expert trace: {log_path}')
    start = np.array([float(result['initial_cube_x_m']),
                      float(result['initial_cube_y_m']), float(first['cube_z_m'])], dtype=float)
    target_xy = np.array([float(result['target_cube_x_m']), float(result['target_cube_y_m'])], dtype=float)
    history = expert_logs_to_samples(log_path, start, rules)
    if not keep_logs:
        log_path.unlink(missing_ok=True)
    lift, transfer = audit_events(history, rules)
    post = post_observe(s4.model, s4.data, ids, start, target_xy, rules.post_observation_s)
    common = audit_final(post, lift, transfer, rules)
    task_t = sum(float(getattr(s4.s3, name)) for name in (
        'APPROACH_TIME', 'DESCEND_TIME', 'CLOSE_TIME', 'LIFT_TIME', 'HOLD_TIME',
        'TRANSFER_TIME', 'PLACE_TIME', 'RELEASE_TIME', 'RETREAT_TIME', 'FINAL_SETTLE_TIME'
    ))
    # If expert aborted early, use actual elapsed time minus its pre-task settle.
    task_t = min(task_t, max(0.0, float(s4.data.time) - rules.post_observation_s - float(s4.s3.SETTLE_BEFORE_GRASP)))
    return {
        'original_lift': bool(result['grasp_success']),
        'original_transfer': bool(result['transfer_success']),
        'original_place': bool(result['place_success']),
        'task_time_s': round(task_t, 4),
        'origin_start_xy': [float(result['initial_cube_x_m']), float(result['initial_cube_y_m'])],
        **common,
    }


def evaluate_sac(env, model, ids, offset, seed, rules):
    obs, info = env.reset(seed=seed, options={'xy_offset': np.asarray(offset, dtype=float)})
    start = env.cube_start.copy()
    target_xy = env.target_xy.copy()
    history = []
    term = trunc = False
    for i in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        history.append(sample_state(
            env.model, env.data, ids=ids, cube_initial_xyz=start,
            target_xy=target_xy, time_s=float(env.data.time)
        ))
        if term or trunc:
            break
    lift, transfer = audit_events(history, rules)
    post = post_observe(env.model, env.data, ids, start, target_xy, rules.post_observation_s)
    common = audit_final(post, lift, transfer, rules)
    return {
        'original_lift': bool(info['has_lifted']),
        'original_transfer': bool(info['has_transferred']),
        'original_place': bool(info['is_success']),
        'task_time_s': round((i+1) * env.control_dt, 4),
        'origin_start_xy': start[:2].tolist(),
        **common,
    }


def plot_results(rows: list[dict], out: Path, rules: BenchmarkRules):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ids = sorted({r['test_index'] for r in rows})
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in ('expert', 'sac'):
        arr = [next(r['final_xy_error_mm'] for r in rows if r['test_index'] == i and r['method'] == label) for i in ids]
        ax.plot(ids, arr, marker='.', linewidth=1, label=label)
    ax.axhline(rules.xy_tol_mm, ls='--', color='black', label=f'{rules.xy_tol_mm:g} mm limit')
    ax.set(xlabel='Paired test case index', ylabel='Final cube XY error (mm)', title='Unified physical Pick-and-Place benchmark')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out/'paired_xy_error.png', dpi=170); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='results_pickplace/seed_42/best_model.zip')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=40000, help='New held-out random offset seed, not Stage4 seed 42')
    p.add_argument('--out', default='results_benchmark/pilot')
    p.add_argument('--max-episode-steps', type=int, default=600)
    p.add_argument('--range-m', type=float, default=0.04)
    p.add_argument('--keep-expert-logs', action='store_true')
    args = p.parse_args()
    if args.episodes <= 0 or args.max_episode_steps <= 0 or args.range_m < 0:
        p.error('Invalid positive episode count or range')
    checkpoint = Path(args.model)
    if not checkpoint.is_absolute(): checkpoint = HERE / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file(): p.error(f'Cannot find checkpoint: {checkpoint}')
    out = Path(args.out)
    if not out.is_absolute(): out = HERE / out
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f'Output not empty: {out}. Choose new --out to avoid mixing tests.')
    out.mkdir(parents=True, exist_ok=True)
    logs_dir = out / '_expert_logs'
    logs_dir.mkdir(exist_ok=True)

    env = PandaPickPlaceEnv(task='pickplace', random_xy_range_m=args.range_m,
                            max_episode_steps=args.max_episode_steps)
    try:
        s4 = env.s4
        s4.OUT = logs_dir
        ids = ids_for(s4)
        model = SAC.load(str(checkpoint), device='cpu')
        if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
            raise RuntimeError('Policy obs/action space mismatch')
        rng = np.random.default_rng(args.seed)
        offsets = rng.uniform(-args.range_m, args.range_m, size=(args.episodes, 2))
        (out/'paired_offsets.json').write_text(json.dumps(offsets.tolist(), indent=2), encoding='utf-8')
        rules = BenchmarkRules()
        (out/'benchmark_rules.json').write_text(json.dumps(asdict(rules), indent=2), encoding='utf-8')
        rows=[]
        for k, off in enumerate(offsets):
            # Stage4 uses its own MjData; Env uses a separate MjData. Both share exact model.
            expert = evaluate_expert(s4, ids, off, k+1, rules, logs_dir, args.keep_expert_logs)
            sac = evaluate_sac(env, model, ids, off, args.seed+k, rules)
            distance_mm = 1000 * float(np.linalg.norm(np.array(expert['origin_start_xy'])-np.array(sac['origin_start_xy'])))
            if distance_mm > 0.1:
                raise RuntimeError(f'Initial settled cube XY mismatch {distance_mm:.4f} mm on case {k+1}')
            for method, r in (('expert',expert), ('sac',sac)):
                rows.append({'test_index':k+1, 'method':method,
                    'offset_x_m':float(off[0]), 'offset_y_m':float(off[1]),
                    **{kk:vv for kk,vv in r.items() if kk != 'origin_start_xy'}})
            write_csv(out/'episodes.csv', rows)
            print(f'[{k+1:03d}/{args.episodes}] Expert={expert["unified_success"]} '
                  f'({expert["final_xy_error_mm"]:.2f} mm) | SAC={sac["unified_success"]} '
                  f'({sac["final_xy_error_mm"]:.2f} mm) | initialXY delta={distance_mm:.4f} mm',flush=True)
        summary = summarize(rows, rules)
        summary['checkpoint'] = str(checkpoint)
        summary['offset_seed'] = args.seed
        summary['range_m'] = args.range_m
        summary['sac_max_episode_steps'] = args.max_episode_steps
        summary['post_intervention'] = 'Hold the last joint and gripper commands for 1s; no policy/expert action, no teleport.'
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        plot_results(rows, out, rules)
        if not args.keep_expert_logs:
            try: logs_dir.rmdir()
            except OSError: pass
        print('\n' + json.dumps(summary, ensure_ascii=False, indent=2))
        print('Saved benchmark:',out)
    finally:
        env.close()


if __name__ == '__main__':
    main()
