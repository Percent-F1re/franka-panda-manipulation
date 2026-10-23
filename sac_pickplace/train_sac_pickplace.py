"""Fine-tune a pretrained SAC Grasp-and-Lift policy on physical Pick-and-Place.

Place next to panda_pickplace_env.py in robot_simulation/sac_pickplace/.

Example (PowerShell, from sac_pickplace directory):
  python train_sac_pickplace.py --steps 10000 --eval-every 5000 --eval-episodes 5 --out results_pickplace/pilot_42
  python train_sac_pickplace.py --steps 500000 --seed 42 --eval-every 25000 --eval-episodes 20

The original Grasp-and-Lift checkpoint is READ ONLY and is NEVER overwritten.
The environment is the user's actual Stage1-4 physics-based PandaPickPlaceEnv,
not scripted expert rollouts. The SAC actor chooses each Cartesian displacement
and gripper command while 6D DLS executes joint-position control.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from panda_pickplace_env import PandaPickPlaceEnv

ROOT = Path(__file__).resolve().parent


def csv_append(path: Path, values: list, header: list | None = None):
    new = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if new and header:
            writer.writerow(header)
        writer.writerow(values)


def evaluate_policy(model: SAC, env: PandaPickPlaceEnv, n_episodes: int,
                    first_seed: int, return_episodes: bool = False):
    """Deterministic evaluation: separate env, fixed target seeds, no training."""
    rows = []
    for k in range(n_episodes):
        seed = first_seed + k
        obs, info = env.reset(seed=seed)
        episode_return = 0.0
        max_rise = float(info['cube_rise_mm'])
        closest = float('inf')
        saw_grasp = False
        max_xy_err = 0.0
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            max_rise = max(max_rise, float(info['cube_rise_mm']))
            closest = min(closest, float(info['grasp_distance_mm']))
            max_xy_err = max(max_xy_err, float(info['cube_xy_error_mm']))
            saw_grasp = saw_grasp or bool(info['grasped'])
            if terminated or truncated:
                break
        else:
            raise RuntimeError('Episode failed to terminate at max_episode_steps')
        row = {
            'seed': seed,
            'grasp_contact_ever': saw_grasp,
            'lift_success': bool(info['has_lifted']),
            'transfer_success': bool(info['has_transferred']),
            'place_success': bool(info['is_success']),
            'steps': step + 1,
            'return': episode_return,
            'max_cube_rise_mm': max_rise,
            'closest_grasp_distance_mm': closest,
            'final_cube_xy_error_mm': float(info['cube_xy_error_mm']),
            'final_cube_speed_m_s': float(info['cube_speed_m_s']),
            'joint_clip_steps': int(info['joint_clip_steps']),
        }
        rows.append(row)

    result = {
        'contact_rate': float(np.mean([r['grasp_contact_ever'] for r in rows])),
        'lift_rate': float(np.mean([r['lift_success'] for r in rows])),
        'transfer_rate': float(np.mean([r['transfer_success'] for r in rows])),
        'place_rate': float(np.mean([r['place_success'] for r in rows])),
        'mean_return': float(np.mean([r['return'] for r in rows])),
        'mean_steps': float(np.mean([r['steps'] for r in rows])),
        'mean_max_cube_rise_mm': float(np.mean([r['max_cube_rise_mm'] for r in rows])),
        'mean_final_xy_error_mm': float(np.mean([r['final_cube_xy_error_mm'] for r in rows])),
    }
    return (result, rows) if return_episodes else result


class PickPlaceEvalCallback(BaseCallback):
    """Select best checkpoint by physical success, not shaped reward alone.

    Priority: Place success > Transfer success > Lift success > mean return.
    """

    def __init__(self, eval_env: PandaPickPlaceEnv, output: Path,
                 interval: int, n_episodes: int, seed: int):
        super().__init__()
        self.eval_env = eval_env
        self.output = output
        self.interval = interval
        self.n_episodes = n_episodes
        self.seed = seed
        self.best_score = (-1.0, -1.0, -1.0, -float('inf'))
        self.history_file = output / 'validation_history.csv'
        self.header = [
            'timesteps', 'place_rate', 'transfer_rate', 'lift_rate',
            'contact_rate', 'mean_return', 'mean_steps',
            'mean_max_cube_rise_mm', 'mean_final_xy_error_mm',
        ]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.interval != 0:
            return True
        report = evaluate_policy(
            self.model, self.eval_env, self.n_episodes, self.seed
        )
        csv_append(
            self.history_file,
            [self.num_timesteps] + [report[key] for key in self.header[1:]],
            header=self.header,
        )
        print(
            f'\n[Validation {self.num_timesteps:,} steps] '
            f'place={report["place_rate"]:.1%}  '
            f'transfer={report["transfer_rate"]:.1%}  '
            f'lift={report["lift_rate"]:.1%}  '
            f'mean_return={report["mean_return"]:.2f}', flush=True,
        )
        score = (
            report['place_rate'], report['transfer_rate'],
            report['lift_rate'], report['mean_return'],
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.output / 'best_model'))
            (self.output / 'best_validation.json').write_text(
                json.dumps({'timesteps': self.num_timesteps, **report},
                           indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            print('  -> Saved best_model.zip (based on staged physical success).', flush=True)
        return True


def make_plot(history_path: Path, figure_path: Path):
    """Optional training validation curve (no dependency on pandas)."""
    if not history_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
        with history_path.open('r', newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        x = np.array([int(r['timesteps']) for r in rows])
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in [
            ('lift_rate', 'Lift'),
            ('transfer_rate', 'Transfer'),
            ('place_rate', 'Place'),
        ]:
            ax.plot(x, [float(r[key]) * 100 for r in rows],
                    marker='o', label=label)
        ax.set(xlabel='SAC fine-tuning timesteps', ylabel='Validation success rate (%)',
               title='Panda Pick-and-Place: validation progress')
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
    except ImportError:
        print('matplotlib unavailable; skipping validation plot.')


def get_parser():
    p = argparse.ArgumentParser(description='Fine-tune SAC Lift -> physical Pick-and-Place')
    p.add_argument('--pretrained', default='results_grasp/seed_42/best_model.zip',
                   help='Existing successful SAC Grasp-and-Lift checkpoint (read only).')
    p.add_argument('--steps', type=int, default=500000, help='NEW training steps in pickplace')
    p.add_argument('--seed', type=int, default=42, help='Training RNG seed')
    p.add_argument('--max-episode-steps', type=int, default=600,
                   help='600 * 0.04s = approx 24 seconds of simulated control')
    p.add_argument('--eval-every', type=int, default=25000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--eval-seed', type=int, default=10000,
                   help='Validation target seeds; keep separate from final held-out tests.')
    p.add_argument('--learning-starts', type=int, default=1000,
                   help='New-task replay warmup: SB3 uses random actions before this many steps')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--out', default=None,
                   help='New output folder, default results_pickplace/seed_<seed>.')
    p.add_argument('--no-baseline', action='store_true',
                   help='Skip initial evaluation of pretrained Lift model on PickPlace.')
    return p


def main():
    p = get_parser()
    args = p.parse_args()
    if args.steps <= 0 or args.max_episode_steps <= 0:
        p.error('--steps and --max-episode-steps must be positive')
    if args.eval_every <= 0 or args.eval_episodes <= 0:
        p.error('--eval-every and --eval-episodes must be positive')
    if args.learning_starts < 0 or args.learning_starts > args.steps:
        p.error('--learning-starts must be >=0 and <= --steps')

    source_path = Path(args.pretrained)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        p.error(f'Cannot find pretrained Lift checkpoint: {source_path}')

    output = (Path(args.out).resolve() if args.out else
              ROOT / 'results_pickplace' / f'seed_{args.seed}')
    # Prevent accidental overwrite of prior checkpoints/results.
    if output.exists() and any(output.iterdir()):
        p.error(f'Output directory is NOT empty: {output}. '
                'Choose a new --out, e.g. results_pickplace/pilot_42.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=True)
    config = {
        **vars(args),
        'source_checkpoint': str(source_path),
        'output_directory': str(output),
        'task': 'pickplace',
        'observations': '46D simulator state from PandaPickPlaceEnv',
        'actions': '4D Cartesian increment + gripper target, DLS low-level control',
        'replay_buffer': 'NEW empty buffer; old Lift replay not loaded',
        'note': 'First 1000 (or user-set learning-starts) actions are random SB3 exploration',
    }
    (output / 'config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    train_env = None
    eval_env = None
    try:
        train_env = Monitor(
            PandaPickPlaceEnv(task='pickplace', max_episode_steps=args.max_episode_steps),
            filename=str(output / 'monitor.csv'),
            info_keywords=('is_success', 'has_lifted', 'has_transferred'),
        )
        eval_env = PandaPickPlaceEnv(task='pickplace',
                                    max_episode_steps=args.max_episode_steps)
        train_env.reset(seed=args.seed)
        model = SAC.load(str(source_path), env=train_env, device=args.device)
        if model.observation_space.shape != train_env.observation_space.shape:
            raise ValueError('Pretrained checkpoint observation shape differs from PickPlace')
        if model.action_space.shape != train_env.action_space.shape:
            raise ValueError('Pretrained checkpoint action shape differs from PickPlace')
        if train_env.observation_space.shape != (46,) or train_env.action_space.shape != (4,):
            raise ValueError('Unexpected environment shape; expected 46D obs / 4D action')

        model.set_random_seed(args.seed)
        # Crucial: the Lift replay buffer is NOT part of model.save().
        # Reset timesteps for this new task so learning_starts warms a fresh buffer.
        model.learning_starts = args.learning_starts

        print('\n===== SAC Lift -> Pick-and-Place fine-tuning =====')
        print('Lift source:', source_path)
        print('Saving NEW run to:', output)
        print('obs:', train_env.observation_space.shape,
              'action:', train_env.action_space.shape)
        print('control step:', eval_env.control_dt, 's')
        print('max episode:', args.max_episode_steps, 'steps')
        print('new replay buffer; will not overwrite Lift weights')

        if not args.no_baseline:
            initial, _ = evaluate_policy(
                model, eval_env, args.eval_episodes, args.eval_seed, return_episodes=True
            )
            (output / 'pretrain_baseline.json').write_text(
                json.dumps(initial, indent=2, ensure_ascii=False), encoding='utf-8'
            )
            print('Initial Lift policy on new PickPlace task:',
                  f'lift={initial["lift_rate"]:.1%}, '
                  f'transfer={initial["transfer_rate"]:.1%}, '
                  f'place={initial["place_rate"]:.1%}', flush=True)

        callback = PickPlaceEvalCallback(
            eval_env, output, args.eval_every, args.eval_episodes, args.eval_seed
        )
        checkpoint = CheckpointCallback(
            save_freq=args.eval_every,
            save_path=str(output / 'checkpoints'),
            name_prefix='sac_pickplace',
        )
        model.learn(
            total_timesteps=args.steps,
            callback=CallbackList([callback, checkpoint]),
            # Resets old Lift step counter; source policy weights are preserved.
            reset_num_timesteps=True,
            progress_bar=False,
        )
        model.save(str(output / 'final_model'))
        make_plot(callback.history_file, output / 'validation_curve.png')
        print('\nFine-tuning complete.')
        print('Final model:', output / 'final_model.zip')
        print('Best model: ', output / 'best_model.zip')
        print('History:    ', callback.history_file)
        if args.steps < args.eval_every:
            print('WARNING: no validation yet. Use final_model.zip for pilot testing; '
                  'best_model.zip will not exist until validation runs.')
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == '__main__':
    main()
