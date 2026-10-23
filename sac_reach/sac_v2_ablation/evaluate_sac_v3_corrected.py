"""Evaluate fixed V3 model, explicitly including all failed Hold samples.

Usage:
    python evaluate_sac_v3_corrected.py --episodes 50 --seed 2027 --train-seed 42
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from panda_reach_env_v3_corrected import PandaReachEnvV3Corrected


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--hold-sec', type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be >= 1')
    here = Path(__file__).resolve().parent
    weight = here / 'results_v3_corrected' / f'seed_{args.train_seed}' / 'sac_panda_reach_v3_corrected.zip'
    if not weight.exists():
        raise FileNotFoundError(f'No weights found: {weight}')
    model = SAC.load(str(weight), device='cpu')
    env = PandaReachEnvV3Corrected(hold_seconds=args.hold_sec)
    out = here / 'results_v3_corrected' / f'seed_{args.train_seed}' / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    summaries, traces = [], []
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            obs, _ = env.reset(seed=seed)
            reached = False
            terminated = False
            distances, hold_distances = [], []
            max_hold_count = 0
            hold_entries = 0
            hold_failures = 0
            actions_during_hold = []
            for step in range(env.max_episode_steps):
                # Capture phase BEFORE stepping: the step that fails Hold
                # may switch phase to Reach before returning info.
                was_holding = env.phase == 1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                dist = float(info['distance_mm'])
                distances.append(dist)
                reached |= bool(info['reach_triggered'])
                hold_entries += int(info['reach_triggered'])
                if was_holding:
                    hold_distances.append(dist)
                    actions_during_hold.append(float(np.linalg.norm(action)))
                    if info['phase'] == 'reach':
                        hold_failures += 1
                max_hold_count = max(max_hold_count, int(info['hold_count']))
                traces.append({
                    'seed': seed, 'step': step + 1,
                    'time_s': round((step + 1) * env.control_dt, 6),
                    'distance_mm': dist, 'was_holding': was_holding,
                    'phase_after': info['phase'],
                    'hold_count': info['hold_count'],
                    'reach_triggered': info['reach_triggered'],
                    'action_l2': float(np.linalg.norm(action)),
                    'reward': float(reward),
                })
                if terminated or truncated:
                    break
            summary = {
                'seed': seed, 'reach_ever': reached,
                'hold_success': bool(terminated),
                'final_mm': distances[-1], 'min_mm': min(distances),
                'max_hold_mm_including_failures': max(hold_distances) if hold_distances else '',
                'max_hold_count': max_hold_count,
                'hold_entries': hold_entries, 'hold_failures': hold_failures,
                'hold_violating_steps': sum(d > 5.0 for d in hold_distances),
                'mean_hold_action_l2': float(np.mean(actions_during_hold)) if actions_during_hold else '',
                'steps': len(distances),
            }
            summaries.append(summary)
            print(f'{i + 1:02d}/{args.episodes} seed={seed} '
                  f'reach={reached} strict_hold={terminated} '
                  f'best_streak={max_hold_count}/{env.hold_steps} '
                  f'fails={hold_failures} final={distances[-1]:.2f} mm')
    finally:
        env.close()

    save_csv(out / 'summary.csv', summaries)
    save_csv(out / 'trace.csv', traces)
    successes = sum(int(r['hold_success']) for r in summaries)
    print(f'Strict Reach + Hold: {successes}/{len(summaries)} ({100 * successes / len(summaries):.1f}%)')
    print(f'Mean max hold streak: {np.mean([r["max_hold_count"] for r in summaries]):.2f} steps')
    print('Results:', out)
    plt.figure(figsize=(10, 5))
    for seed in range(args.seed, args.seed + args.episodes):
        sub = [r for r in traces if r['seed'] == seed]
        plt.plot([r['time_s'] for r in sub], [r['distance_mm'] for r in sub], alpha=0.25)
    plt.axhline(5.0, linestyle='--', color='black', label='5 mm threshold')
    plt.xlabel('Simulation time (s)')
    plt.ylabel('Position error (mm)')
    plt.title('SAC-v3 corrected Reach + Hold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'trajectories.png', dpi=180)
    plt.close()


if __name__ == '__main__':
    main()
