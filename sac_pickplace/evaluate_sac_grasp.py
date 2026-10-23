"""Deterministic SAC lift policy evaluation with episode CSV and trace plot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from panda_pickplace_env import PandaPickPlaceEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='results_grasp/seed_42/final_model.zip')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--max-episode-steps', type=int, default=350)
    p.add_argument('--trace-seed', type=int, default=2060,
                   help='Seed to plot (must be in the evaluated seed range)')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.is_file():
        p.error(f'Cannot find trained model: {model_path}')
    out = Path(args.out).resolve() if args.out else root / 'results_grasp' / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    env = PandaPickPlaceEnv(task='lift', max_episode_steps=args.max_episode_steps)
    try:
        model = SAC.load(str(model_path), device='cpu')
        if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
            raise ValueError('Model/environment observation or action shape differs')
        summaries, traces = [], []
        for i in range(args.episodes):
            episode_seed = args.seed + i
            obs, info = env.reset(seed=episode_seed)
            episode_return, max_rise, closest_grasp = 0.0, -float('inf'), float('inf')
            seen_bilateral = False
            for k in range(args.max_episode_steps):
                act, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(act)
                episode_return += float(reward)
                max_rise = max(max_rise, float(info['cube_rise_mm']))
                closest_grasp = min(closest_grasp, float(info['grasp_distance_mm']))
                table, left, right = info['contacts']
                seen_bilateral = seen_bilateral or bool(left > 0 and right > 0)
                if episode_seed == args.trace_seed:
                    traces.append(dict(
                        step=k+1, sim_time_s=(k+1)*env.control_dt,
                        grasp_distance_mm=float(info['grasp_distance_mm']),
                        cube_rise_mm=float(info['cube_rise_mm']),
                        table_contact=int(table), left_contact=int(left),
                        right_contact=int(right),
                        grip_action=float(act[3]),
                        reward=float(reward),
                    ))
                if terminated or truncated:
                    break
            row = dict(seed=episode_seed, success=bool(info['is_success']),
                       steps=k+1, sim_time_s=(k+1)*env.control_dt,
                       total_reward=episode_return,
                       max_cube_rise_mm=max_rise,
                       min_grasp_distance_mm=closest_grasp,
                       ever_bilateral_contact=seen_bilateral,
                       joint_clip_steps=info['joint_clip_steps'])
            summaries.append(row)
            print(f'{i+1:02d}/{args.episodes} seed={episode_seed} '
                  f'success={row["success"]} max_rise={max_rise:.1f} mm '
                  f'closest={closest_grasp:.1f} mm steps={k+1}', flush=True)

        def save_csv(filename, rows):
            if not rows:
                return
            with (out / filename).open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        save_csv('episodes.csv', summaries)
        save_csv('trace.csv', traces)
        success_rate = float(np.mean([r['success'] for r in summaries]))
        stats = {
            'model': str(model_path), 'episodes': len(summaries),
            'test_seed_start': args.seed, 'success_count': int(sum(r['success'] for r in summaries)),
            'success_rate': success_rate,
            'mean_max_cube_rise_mm': float(np.mean([r['max_cube_rise_mm'] for r in summaries])),
            'mean_min_grasp_distance_mm': float(np.mean([r['min_grasp_distance_mm'] for r in summaries])),
        }
        (out / 'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        print('\n', json.dumps(stats, indent=2))
        if traces:
            import matplotlib.pyplot as plt
            times = [r['sim_time_s'] for r in traces]
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            axes[0].plot(times, [r['grasp_distance_mm'] for r in traces])
            axes[0].set_ylabel('Grasp-point distance (mm)')
            axes[1].plot(times, [r['cube_rise_mm'] for r in traces])
            axes[1].axhline(40.0, ls='--', color='black', label='40 mm lift criterion')
            axes[1].set_ylabel('Cube rise (mm)')
            axes[1].set_xlabel('Simulation time (s)')
            axes[1].legend()
            for ax in axes:
                ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / 'seed_trace.png', dpi=160)
            plt.close(fig)
        print('Saved:', out)
    finally:
        env.close()


if __name__ == '__main__':
    main()
