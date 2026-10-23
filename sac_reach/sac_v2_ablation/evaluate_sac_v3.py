"""Evaluate strict uninterrupted 1-second holding with SAC-v3.

CSV contains all 50 seed outcomes; trajectories are recorded for plotting.
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from panda_reach_env_v3 import PandaReachEnvV3


def save_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--train-seed', type=int, default=42)
    p.add_argument('--hold-sec', type=float, default=1.0)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    weight = here / 'results_v3' / f'seed_{args.train_seed}' / 'sac_panda_reach_v3.zip'
    if not weight.exists():
        raise FileNotFoundError(weight)
    model = SAC.load(str(weight), device='cpu')
    env = PandaReachEnvV3(hold_seconds=args.hold_sec)
    out = here / 'results_v3' / f'seed_{args.train_seed}' / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    summaries, traces = [], []
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            obs, _ = env.reset(seed=seed)
            reached = False
            done = False
            distances = []
            hold_distances = []
            for step in range(env.max_episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                dist = float(info['distance_mm'])
                distances.append(dist)
                if info['reach_triggered']:
                    reached = True
                if info['phase'] == 'hold' and not info['reach_triggered']:
                    hold_distances.append(dist)
                traces.append({'seed': seed, 'step': step + 1,
                               'time_s': (step + 1) * env.control_dt,
                               'distance_mm': dist, 'phase': info['phase'],
                               'hold_count': info['hold_count']})
                if done or truncated:
                    break
            row = {'seed': seed, 'reach_ever': reached,
                   'hold_success': bool(done), 'final_mm': distances[-1],
                   'min_mm': min(distances), 'max_hold_mm':
                   max(hold_distances) if hold_distances else '',
                   'steps': len(distances)}
            summaries.append(row)
            print(f"{i+1:02d}/{args.episodes} seed={seed} "
                  f"hold={done} final={distances[-1]:.2f} mm")
    finally:
        env.close()
    save_csv(out / 'summary.csv', summaries, list(summaries[0]))
    save_csv(out / 'trace.csv', traces, list(traces[0]))
    success = sum(r['hold_success'] for r in summaries)
    print(f'Strict Reach + Hold: {success}/{args.episodes} ({success/args.episodes:.1%})')
    plt.figure(figsize=(10, 5))
    for seed in range(args.seed, args.seed + args.episodes):
        rows = [r for r in traces if r['seed'] == seed]
        plt.plot([r['time_s'] for r in rows], [r['distance_mm'] for r in rows], alpha=.25)
    plt.axhline(5, color='black', linestyle='--', label='5 mm')
    plt.xlabel('Time (s)')
    plt.ylabel('Position error (mm)')
    plt.title('SAC-v3 Reach + Hold evaluation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'trajectories.png', dpi=180)
    plt.close()
    print('Saved:', out)

if __name__ == '__main__':
    main()
