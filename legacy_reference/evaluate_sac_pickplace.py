"""Held-out evaluation for a physical SAC Pick-and-Place checkpoint.

Usage:
  python evaluate_sac_pickplace.py --model results_pickplace/seed_42/best_model.zip --episodes 50 --seed 30000

The task completes ONLY when PandaPickPlaceEnv returns is_success from physical
lift, transfer, release and settled placement checks. No expert trajectory is used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from panda_pickplace_env import PandaPickPlaceEnv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results_pickplace/seed_42/best_model.zip')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=30000)
    parser.add_argument('--max-episode-steps', type=int, default=600)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be >= 1')

    checkpoint = Path(args.model)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Model not found: {checkpoint}')

    out = (Path(args.out).resolve() if args.out else ROOT / 'results_pickplace' / 'test_heldout')
    out.mkdir(parents=True, exist_ok=True)

    env = PandaPickPlaceEnv(task='pickplace', max_episode_steps=args.max_episode_steps)
    try:
        model = SAC.load(str(checkpoint), device='cpu')
        if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
            raise ValueError('Incompatible pretrained observation/action spaces')
        rows = []
        for idx in range(args.episodes):
            ep_seed = args.seed + idx
            obs, info = env.reset(seed=ep_seed)
            total_reward = 0.0
            max_rise = float(info['cube_rise_mm'])
            ever_contact = False
            for t in range(args.max_episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                max_rise = max(max_rise, float(info['cube_rise_mm']))
                ever_contact = ever_contact or bool(info['grasped'])
                if terminated or truncated:
                    break
            row = {
                'seed': ep_seed,
                'contact_ever': ever_contact,
                'lift_success': bool(info['has_lifted']),
                'transfer_success': bool(info['has_transferred']),
                'place_success': bool(info['is_success']),
                'steps': t + 1,
                'sim_time_s': float(info['sim_time_s']),
                'episode_return': total_reward,
                'max_cube_rise_mm': max_rise,
                'final_cube_xy_error_mm': float(info['cube_xy_error_mm']),
                'final_cube_speed_m_s': float(info['cube_speed_m_s']),
                'joint_clip_steps': int(info['joint_clip_steps']),
            }
            rows.append(row)
            print(f'{idx+1:02d}/{args.episodes} seed={ep_seed} '
                  f'lift={row["lift_success"]} transfer={row["transfer_success"]} '
                  f'place={row["place_success"]} '
                  f'xy_error={row["final_cube_xy_error_mm"]:.1f}mm '
                  f'steps={row["steps"]}', flush=True)

        with (out / 'episodes.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            'checkpoint': str(checkpoint),
            'episodes': len(rows),
            'first_test_seed': args.seed,
            'contact_rate': float(np.mean([r['contact_ever'] for r in rows])),
            'lift_rate': float(np.mean([r['lift_success'] for r in rows])),
            'transfer_rate': float(np.mean([r['transfer_success'] for r in rows])),
            'place_rate': float(np.mean([r['place_success'] for r in rows])),
            'mean_episode_steps': float(np.mean([r['steps'] for r in rows])),
            'mean_final_cube_xy_error_mm_all': float(np.mean([r['final_cube_xy_error_mm'] for r in rows])),
            'mean_final_cube_xy_error_mm_successes': (
                float(np.mean([r['final_cube_xy_error_mm'] for r in rows if r['place_success']]))
                if any(r['place_success'] for r in rows) else None
            ),
        }
        (out / 'summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print('\n', json.dumps(summary, indent=2, ensure_ascii=False))
        print('Saved to', out)
    finally:
        env.close()


if __name__ == '__main__':
    main()
