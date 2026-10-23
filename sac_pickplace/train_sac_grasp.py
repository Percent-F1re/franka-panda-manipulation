"""Train state-based SAC Grasp-and-Lift in the Stage4-derived Panda environment.

Place next to panda_pickplace_env.py in robot_simulation/sac_pickplace/.
No expert stage commands are used by the SAC policy.
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


class GraspEvalCallback(BaseCallback):
    """Evaluate on fixed, independently seeded target positions.

    Save best model using success rate FIRST, evaluation return as tie-breaker.
    This avoids optimizing only the shaped reward in model selection.
    """

    def __init__(self, env, out_dir: Path, eval_every: int = 25000,
                 episodes: int = 20, seed: int = 20000):
        super().__init__()
        self.env = env
        self.out_dir = out_dir
        self.eval_every = int(eval_every)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.best = (-1.0, -float('inf'))
        self.csv_path = out_dir / 'validation_history.csv'
        with self.csv_path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'timesteps', 'success_rate', 'mean_return', 'mean_steps',
                'mean_max_cube_rise_mm',
            ])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every:
            return True
        successes, returns, steps, max_rises = [], [], [], []
        for i in range(self.episodes):
            obs, _ = self.env.reset(seed=self.seed + i)
            episode_return, top_rise = 0.0, -float('inf')
            for step in range(self.env.max_episode_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_return += float(reward)
                top_rise = max(top_rise, float(info['cube_rise_mm']))
                if terminated or truncated:
                    break
            successes.append(float(info['is_success']))
            returns.append(episode_return)
            steps.append(step + 1)
            max_rises.append(top_rise)

        success_rate = float(np.mean(successes))
        mean_return = float(np.mean(returns))
        with self.csv_path.open('a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                self.num_timesteps, success_rate, mean_return,
                float(np.mean(steps)), float(np.mean(max_rises))
            ])
        print(f'\nValidation @ {self.num_timesteps:,} steps: '
              f'success={success_rate:.1%}, mean_return={mean_return:.2f}, '
              f'mean_max_rise={np.mean(max_rises):.1f}mm', flush=True)
        score = (success_rate, mean_return)
        if score > self.best:
            self.best = score
            self.model.save(str(self.out_dir / 'best_model'))
            print('Saved validation best model.', flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-episode-steps', type=int, default=350)
    p.add_argument('--eval-every', type=int, default=25000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--learning-starts', type=int, default=5000)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--out', default=None,
                   help='Optional output directory; default results_grasp/seed_N')
    args = p.parse_args()
    if args.steps <= 0 or args.max_episode_steps <= 0 or args.eval_every <= 0:
        p.error('steps, max-episode-steps and eval-every must be > 0')
    root = Path(__file__).resolve().parent
    out_dir = (Path(args.out).resolve() if args.out else
               root / 'results_grasp' / f'seed_{args.seed}')
    if out_dir.exists() and any(out_dir.glob('*.zip')):
        p.error(f'{out_dir} already has saved weights; use a NEW --out directory '
                'so that training runs are not silently overwritten')
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'output_directory': str(out_dir)}
    (out_dir / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    train_env = Monitor(
        PandaPickPlaceEnv(task='lift', max_episode_steps=args.max_episode_steps),
        filename=str(out_dir / 'monitor.csv'),
        info_keywords=('is_success', 'has_lifted'),
    )
    eval_env = PandaPickPlaceEnv(task='lift', max_episode_steps=args.max_episode_steps)
    try:
        train_env.reset(seed=args.seed)
        model = SAC(
            'MlpPolicy', train_env, seed=args.seed,
            learning_rate=3e-4, buffer_size=200000,
            learning_starts=args.learning_starts, batch_size=256,
            tau=0.005, gamma=0.99, train_freq=1,
            gradient_steps=1, ent_coef='auto',
            policy_kwargs={'net_arch': [256, 256]},
            verbose=1, device=args.device,
        )
        eval_cb = GraspEvalCallback(eval_env, out_dir,
                                    eval_every=args.eval_every,
                                    episodes=args.eval_episodes)
        checkpoint = CheckpointCallback(
            save_freq=max(1, args.eval_every),
            save_path=str(out_dir / 'checkpoints'),
            name_prefix='sac_grasp',
        )
        print('Observation dimension:', train_env.observation_space.shape)
        print('Action dimension:', train_env.action_space.shape)
        print('Output:', out_dir)
        model.learn(total_timesteps=args.steps,
                    callback=CallbackList([eval_cb, checkpoint]),
                    progress_bar=False)
        model.save(str(out_dir / 'final_model'))
        print('Saved final_model.zip')
        if args.steps < args.eval_every:
            print('NOTE: no validation happened. Run a longer experiment or '
                  'reduce --eval-every; final_model.zip is still saved.')
    finally:
        train_env.close()
        eval_env.close()


if __name__ == '__main__':
    main()
